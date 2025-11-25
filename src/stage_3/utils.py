# src/stage_3/utils.py

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# Temporal smoothing
def apply_temporal_smoothing(pred_df: pd.DataFrame, window: int):
    half = window // 2
    smoothed_groups = []

    for (subject_id, night), group in pred_df.groupby(["subject_id", "night"], sort=False):

        # Create groups of epoch based on subject_id + night combination 
        # and sorted based on epoch_id (which is sorted on epoch onset second)
        group = group.sort_values("epoch_id").copy()
        preds = group["pred_label"].to_numpy()
        preds_len = len(preds)
        smoothed = preds.copy()

        for i in range(preds_len):
            start = max(0, i - half)
            end = min(preds_len, i + half + 1)
            window_labels = preds[start:end]

            vals, counts = np.unique(window_labels, return_counts=True)
            smoothed[i] = vals[np.argmax(counts)]

        group["pred_label_smoothed"] = smoothed
        smoothed_groups.append(group)

    return pd.concat(smoothed_groups).sort_index()