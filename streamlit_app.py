# app/streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    log_loss,
    roc_auc_score,
)

from src.stage_3.utils import apply_temporal_smoothing

PREDICTIONS_PATH = Path("test_predictions_test_metrics.csv") # Path to test set predictions (CSV file)

LABEL_TO_NAME = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"} # Sleep stage mapping

# Load prediction table and cache it
@st.cache_data
def load_predictions():
    df_pred = pd.read_csv(PREDICTIONS_PATH)
    return df_pred


# Build a list of test set recordings
def build_recording_dropdown(df_test: pd.DataFrame):
    recordings = []

    # Group epochs by subject_id + night to get one entry per recording
    for (subject_id, night), group in df_test.groupby(["subject_id", "night"]):
        group = group.sort_values("epoch_id") # Sort all the epochs within a recording by epoch_id

        start_sec = 0.0 # Start time (second) of the first epoch that recording
        end_sec = float(group["epoch_start_point"].iloc[-1] + 30.0) # Get the end time (second) of the last epcoh of that recording and add 30 to get finish time
        duration_hours = (end_sec - start_sec) / 3600.0 # Convert to hours

        label = f"{subject_id} – Night {night} – {duration_hours:.1f} h" # Create label that will appear in recording selection dropdown

        recordings.append({"label": label, "subject_id": subject_id, "night": night}) # Add eac recording object (label, subject_id, night number) in the recordings list

    recordings = sorted(recordings, key=lambda x: (x["subject_id"], x["night"])) # Sort recordings by subject_id for consistent ordering
    return recordings


# Extract a single recording (subject_id + night) from the predictions table
def get_recording_data(df_pred: pd.DataFrame, subject_id: str, night: int):

    df_pred_copy = df_pred[(df_pred["subject_id"] == subject_id) & (df_pred["night"] == night)].copy() # Look into the entire prediction table and filter out rows for a specific recording

    df_pred_copy = df_pred_copy.sort_values("epoch_id").reset_index(drop=True) # Sort epochs in time order (index is in time order)

    y_true = df_pred_copy["true_label"].to_numpy() # Get true label for each epoch from the predictions table (in int)

    y_pred_raw = df_pred_copy["pred_label"].to_numpy() # Get the predicted label for each epoch from the predictions table (in int)

    # Per class probabilities for each epoch from the predictions table (in float)
    probability_columns = [column for column in df_pred_copy.columns if column.startswith("p_")]
    y_prob = df_pred_copy[probability_columns].to_numpy()

    times_sec = df_pred_copy["epoch_start_point"].to_numpy() # Start times of each epoch from the predictions table (seconds since recording start)


    return df_pred_copy, y_true, y_pred_raw, y_prob, times_sec

# Compute sleep report (total recording time, estimated sleep time, sleep efficiency, stage distribution)
def compute_sleep_report(y_pred: np.ndarray):

    total_epochs = len(y_pred) # Total number of epcohs
    minutes_per_epoch = 0.5 # Each epoch is 0.5 mins as it is 30 secs 
    total_minutes = total_epochs * minutes_per_epoch # Total recording time in minutes

    sleep_mask = y_pred != 0 # Mask for all sleep labels (N1, N2, N3, REM). 0 coressponds to Wake
    total_sleep_epochs = sleep_mask.sum() # Total number of all sleep related epochs
    total_sleep_minutes = total_sleep_epochs * minutes_per_epoch # Total duration of sleep time in minutes

    if total_sleep_minutes > 0:
        sleep_efficiency = (total_sleep_minutes / total_minutes) * 100.0 # Calculate sleep efficiency (%)
    else:
        sleep_efficiency = 0.0 # If model only predics Wake then sleep efficiency will be 0

    stage_report = [] # Stage distribution over sleep time (exclude Wake)


    for label, name in LABEL_TO_NAME.items():
        stage_mask = y_pred == label # Filter out all epochs predicted a particular label
        stage_epochs = stage_mask.sum() # Total all epochs predicted a particular label
        stage_minutes = stage_epochs * minutes_per_epoch # Total duration of a particular label

        if label == 0 or total_sleep_minutes == 0: # If epoch is Wake (W) or model has only predicted Wake don't include
            stage_percent_of_sleep = 0.0 # Set as 0%
        else:
            stage_percent_of_sleep = (stage_minutes / total_sleep_minutes) * 100.0 # Percentage of a parituclar label's duration out of total sleep time

        # Add an object (stage name, duration in minutes, percentage of sleep time) for each label in the stage_report list
        stage_report.append({"Stage": name, "Minutes": round(stage_minutes, 1), "% of sleep": round(stage_percent_of_sleep, 1)})

    stage_df = pd.DataFrame(stage_report) # Covert stage distribution to dataframe

    # Compute some sleep related stats
    metrics = {
        "total_recording_hours": round(total_minutes / 60.0, 2), # Total recording duration in hours
        "total_sleep_hours": round(total_sleep_minutes / 60.0, 2), # Total sleep time in hours
        "sleep_efficiency": round(sleep_efficiency, 2) # Sleep efficiency
    }

    return metrics, stage_df # Return all the metrics and the stage distribution as a dataframe

# Compute evaluation metrics
def compute_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probabilities: np.ndarray):

    # Overall metrics to be calculated
    overall_metrics = [
        {"Metric": "Accuracy", "Value": round(accuracy_score(y_true, y_pred), 4)},
        {"Metric": "Macro F1", "Value": round(f1_score(y_true, y_pred, average="macro"), 4)},
        {"Metric": "Weighted F1", "Value": round(f1_score(y_true, y_pred, average="weighted"), 4)},
        {"Metric": "Cohen's Kappa", "Value": round(cohen_kappa_score(y_true, y_pred), 4)},
        {"Metric": "Macro AUC", "Value": round(roc_auc_score(y_true, y_probabilities, multi_class="ovr", average="macro"), 4)},
        {"Metric": "Weighted AUC", "Value": round(roc_auc_score(y_true, y_probabilities, multi_class="ovr", average="weighted"), 4)},
        {"Metric": "Log loss", "Value": round(log_loss(y_true, y_probabilities), 4)}
    ]

    overall_df = pd.DataFrame(overall_metrics) # Convert to dataframe

    # Per class metrics from classification_report
    report = classification_report(y_true, y_pred, labels=list(LABEL_TO_NAME.keys()), output_dict=True, zero_division=0)

    per_class_rows = [] # Empty list to store per class metrics

    # Loop through dict of all labels
    for label, name in LABEL_TO_NAME.items():
        if str(label) in report:
            entry = report[str(label)] # Get entry of that label from the classififcation report
            per_class_rows.append(
                {
                    "Stage": name,
                    "Precision": round(entry["precision"], 4),
                    "Recall": round(entry["recall"], 4),
                    "F1-score": round(entry["f1-score"], 4),
                    "Support": entry["support"],
                }
            )

    per_class_df = pd.DataFrame(per_class_rows) # Convert per class metrics to dataframe

    return overall_df, per_class_df


# Plot hypnogram to show true vs predicted (raw and smoothed)
def plot_hypnogram(duration: np.ndarray, y_true: np.ndarray, y_pred_raw: np.ndarray | None, y_pred_smooth: np.ndarray | None):

    fig, ax = plt.subplots(figsize=(10, 4)) # Set size of plot

    ax.step(duration, y_true, where="post", label="True", linewidth=1.5, color="C0") # Plot true labels

    # Plot raw predictions if provided
    if y_pred_raw is not None:
        ax.step(duration, y_pred_raw, where="post", label="Model Predictions (Raw)", linewidth=1, color="C1")

    # Plot smoothed predictions if provided
    if y_pred_smooth is not None:
        ax.step(duration, y_pred_smooth, where="post", label="Model Predictions (smoothed)", linewidth=1, color="C2")

    ax.set_yticks(list(LABEL_TO_NAME.keys())) # Set ticks at index 0-4
    ax.set_yticklabels(list(LABEL_TO_NAME.values())) # Display ticks as string value of labels
    ax.set_ylim(-0.5, 4.5) # Vertical limits of x-axis. Slightly more for padding

    ax.set_xlabel("Time from recording start (hours)") # X-axis label
    ax.set_ylabel("Sleep stage") # Y-axis label
    ax.set_title("Hypnogram: True vs Model Predictions") # Plot title

    ax.legend(loc="upper right") # Set legend to upper right


    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig


def main():
    st.title("Sleep Stage Classifier") # Page title

    st.subheader("EXPERIMENTAL DEMO (NOT FOR CLINICAL USE)") # DISCLAIMER FOR USER AS THIS ISN'T FOR CLINICAL USE

    # Brief description about the tool
    st.markdown(
        """
        This app demonstrates an experimental **sleep stage classification model** trained on EEG data
        (Fpz-Cz and Pz-Oz channels) from the [Sleep-EDF Database Expanded](https://www.physionet.org/content/sleep-edfx/1.0.0/).

        - A Random Forest classifier was trained on hand-crafted EEG features.
        Because the full model is large, its predictions (labels + probabilities)
        were pre-computed on the test set and stored in a CSV file, which this app
        loads for fast, lightweight visualization.
        - Predictions may be inaccurate.
        - True labels are derived from expert-scored hypnograms in the Sleep-EDF database.
        """
    )

    st.markdown(
    """
        ##### Use the sidebar to:
        - Select a recording (subject + night)
        - Toggle **raw** vs **smoothed** predictions
        - Adjust the **smoothing window**
        - Choose a **time range** to zoom into

        ##### The main page shows:
        - A **sleep report** (recording duration, estimated sleep time, stage distribution)
        - A **hypnogram** with true labels (blue), raw predictions (orange), and smoothed predictions (green)
        - **Model performance** metrics for the full recording

        ##### Temporal smoothing:
        Temporal smoothing helps stabilise the predictions by cleaning up the raw sleep-stage predictions.
        Instead of predicting each 30-second epoch in isolation, the model looks at a small window of neighbouring epochs
        and replaces the final label with the most common (mode) stage within that window.

        This makes the timeline smoother, removes unrealistic one-off spikes (e.g., N2 to N3 to N2 within 3 epochs), and
        produces a more physiologically plausible hypnogram, which is more aligned with how human scorers read sleep.

        **A window size of 5** is used for test-set evaluation, as it yielded the strongest overall performance during cross-validation.
        Full results and methodology are documented on this project's [GitHub](https://github.com/dishant26201/sleep-analytics-pipeline.git)
        """
    )

    st.markdown("---") # Divider

    df_pred = load_predictions()  # Load the prediction table

    recordings = build_recording_dropdown(df_pred) # List of test recordings for the dropdown

    # LOAD RECORDINGS SECTION (SIDEBAR)

    st.sidebar.header("Recording selection") # Recording selection section in the sidebar

    recording_labels = [recording["label"] for recording in recordings]  # List of recording labels

    selected_label = st.sidebar.selectbox("Select a test recording (subject + night)",
        options=recording_labels,
        help="Many subjects were recorded over multiple nights," \
        " so each option represents one specific recording," \
        " lasting approximately 20-24 hours.") # Create dropdown menu to select recording with all recording labels in the dropdown

    selected_record = next(rec for rec in recordings if rec["label"] == selected_label) # Find the recording matching the selected label

    # Extract subject ID and night from the chosen recording
    subject_id = selected_record["subject_id"]
    night = selected_record["night"]

    st.sidebar.markdown("---") # Divider

    # PREDICTION SETTINGS SECTION (SIDEBAR)

    st.sidebar.header("Prediction settings") # Recording predictions settings section in the sidebar

    show_raw = st.sidebar.checkbox("Show raw model predictions", value=False, help="Show per-epoch predictions made by the model.") # Sidebar toggle to display raw model predictions for each epoch
    show_smoothed = st.sidebar.checkbox("Show smoothed predictions", value=True, help="Apply temporal smoothing across epochs.") # Sidebar toggle to display predictions with temporal smoothing applied for each epoch

    # Setup window slider for temporal smoothing
    window_size = st.sidebar.slider("Smoothing window size (epochs)", min_value=3, max_value=15, step=2, value=5,
        help="Window size in epochs for temporal smoothing (each epoch = 30 seconds).")


    df_pred_copy, y_true, y_pred_raw, y_prob, times_sec = get_recording_data(df_pred, subject_id, night) # Load all prediction data for the selected recording

    times_hours = times_sec / 3600.0 # Converts each epoch start time from seconds to hours
    total_hours = (times_hours[-1]) + (30.0 / 3600.0) # Total recording duration in hours

    st.sidebar.markdown("---") # Divider


    # TIME RANGE SECTION (SIDEBAR)

    st.sidebar.header("Time range") # Range slider for time window in sidebar

    # Setup time range (hours)
    start_hour, end_hour = st.sidebar.slider(
        "Select time range for the hypnogram (in hours from start to end of the recording)",
        min_value=0.0, # Start at 0 hours
        max_value=(np.round(total_hours, 1)), # End at the total duration
        value=(0.0, np.round(total_hours, 1)), # Default range
        step=0.5,
    )

    pred_df = df_pred_copy[["subject_id", "night", "epoch_id", "epoch_start_point"]].copy() # Build a copy of the copied predictions table for smoothing and apply mode filter
    pred_df["pred_label"] = y_pred_raw  # Add a column with raw predictions

    pred_df_smoothed = apply_temporal_smoothing(pred_df, window=window_size) # Apply smoothing (mode filter) and capture output df
    y_pred_smooth_full = pred_df_smoothed["pred_label_smoothed"].to_numpy() # Turn the smoothed predictions to a numpy array

    mask_range = (times_hours >= start_hour) & (times_hours <= end_hour) # Mask for selecting epochs within the chosen time range

    times_range = times_hours[mask_range] # Apply mask to only have epochs within the range
    y_true_range = y_true[mask_range] # Apply mask to only have epochs within the range
    y_pred_raw_range = y_pred_raw[mask_range] if show_raw else None # Apply mask to raw predications, to only have epochs within the range 
    y_pred_smooth_range = y_pred_smooth_full[mask_range] if show_smoothed else None # Apply mask to smoothed predications, to only have epochs within the range 


    # SLEEP REPORT SUMMARY (MAIN AREA)

    st.markdown(
        """
        ### Sleep Report Summary

        This section summarizes the sleep patterns for the selected recording based on the model's predictions.
        It reports total sleep time and displays stage distribution, which displays how much time
        the model thinks you spent in each sleep stage. **These values update dynamically
        based on the smoothing window you choose.**

        """
    )

    metrics, stage_df = compute_sleep_report(y_pred_smooth_full) # Compute sleep related metrics on the smoothed predictions

    # Display overall sleep related metrics
    st.markdown(
        f"""
        - **Total recording time:** {metrics["total_recording_hours"]:.2f} hours
        - **Estimated sleep time:** {metrics["total_sleep_hours"]:.2f} hours
        """
    )

    # - **Sleep efficiency:** {metrics["sleep_efficiency"]:.2f}%

    st.dataframe(stage_df, use_container_width=True) # Display distrubution chart (df) in table form

    st.markdown("---") # Divider


    # HYPNOGRAM (MAIN AREA)

    st.markdown(
        """
        ### Hypnogram (True vs Model Predictions)

        This plot compares the expert-scored hypnogram with the model’s predictions over time. 
        You can **zoom into specific periods using the time-range slider** and **choose whether to display 
        raw or smoothed predictions** to see how temporal smoothing affects stage continuity. Once again
        **These plot updates dynamically based on the smoothing window you choose.**

        """
    )

    # Plot hypnogram with true vs selected prediction views
    fig = plot_hypnogram(
        times_range,
        y_true_range,
        y_pred_raw_range,
        y_pred_smooth_range,
    )
    st.pyplot(fig)


    st.markdown("---") # Divider

    # MODEL PERFORMANCE (MAIN AREA)

    st.markdown(
        """
        ### Model Performance

        This section displays the model’s overall performance metrics alongside per-class scores
        for each sleep stage. All values are computed from the smoothed predictions and
        for the **full duration of the selected recording**.

        **The values will update dynamically based on the smoothing window you choose.**

        """
    )


    overall_df, per_class_df = compute_model_metrics(y_true, y_pred_smooth_full, y_prob) # Call function to compute metrics

    # Show overall and per class metrics side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Overall metrics")
        st.dataframe(overall_df, use_container_width=True)

    with col2:
        st.markdown("##### Per-class metrics")
        st.dataframe(per_class_df, use_container_width=True)


if __name__ == "__main__":
    main()
