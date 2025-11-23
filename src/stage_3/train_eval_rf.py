# src/stage_3/train_eval_rf.py

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from .utils import apply_mode_filter
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from .load_data import generate_training_splits


MODEL_VERSION = "rf_cv_v3"
OUTPUT_DIRECTORY = Path("models") / MODEL_VERSION
OUTPUT_DIRECTORY.mkdir(parents = True, exist_ok = True)
WAKE_LABEL = 0
N1_LABEL = 1
N2_LABEL = 2
N3_LABEL = 3
REM_LABEL = 4
LABEL_TO_NAME = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

X_train, y_train, X_cv, y_cv, X_test, y_test, feature_columns, cv_meta, test_meta = generate_training_splits()

count = np.bincount(y_train)
w_count = count[WAKE_LABEL]
n1_count = count[N1_LABEL]
n2_count = count[N2_LABEL]
n3_count = count[N3_LABEL]
rem_count = count[REM_LABEL]

print(f"\nOriginal Count")
print(f"Wake: {w_count}")
print(f"N1: {n1_count}")
print(f"N2: {n2_count}")
print(f"N3: {n3_count}")
print(f"REM: {rem_count}")

w_count = count[WAKE_LABEL]
n1_count = 15 * n1_count
n2_count = 2 * n2_count
n3_count = 5 * n3_count
rem_count = 7 * rem_count


over_sampling_strategy_n1 = {
    1: n1_count
}

over_sampling_strategy_n2_n3_rem = {
    2: n2_count,
    3: n3_count,
    4: rem_count
}

print(f"\nCount after over/under sampling")
print(f"Wake: {w_count}")
print(f"N1: {n1_count}")
print(f"N2: {n2_count}")
print(f"N3: {n3_count}")
print(f"REM: {rem_count}")

smote1 = BorderlineSMOTE(
    sampling_strategy = over_sampling_strategy_n1,
    k_neighbors = 3,
    random_state = 42,
)
X_train_border_smote, y_train_border_smote = smote1.fit_resample(X_train, y_train)


smote2 = SMOTE(
    sampling_strategy = over_sampling_strategy_n2_n3_rem,
    random_state = 42,
    k_neighbors = 5
)
X_train_over, y_train_over = smote2.fit_resample(X_train_border_smote, y_train_border_smote)


# rf_parameters = {
#     "bootstrap": True,
#     "class_weight": "balanced",
#     "criterion": "gini",
#     "max_depth": None,
#     "max_features": "sqrt",
#     "min_samples_leaf": 1,
#     "min_samples_split": 2,
#     "n_estimators": 100,
#     "n_jobs": -1,
#     "random_state": 42
# }

rf_parameters = {
    "bootstrap": True,
    "criterion": "log_loss",
    "n_estimators": 300,
    "max_depth": 25,
    "min_samples_split": 6,
    "min_samples_leaf": 3,
    "max_features": 0.5, 
    "class_weight": {
        0: 1.0,
        1: 3.0,
        2: 1.0,
        3: 1.4,
        4: 2.0,
    },
    "n_jobs": -1,
    "random_state": 42,
}

rf_model = RandomForestClassifier(**rf_parameters)

rf_model.fit(X_train_over, y_train_over)

dump(rf_model, OUTPUT_DIRECTORY / f"{MODEL_VERSION}.joblib")

# rf_model = load("models/rf_cv_v1/rf_cv_v1.joblib")

CLASSES = list(rf_model.classes_)  # 0 = W, 1 = N1, 2 = N2, 3 = N3, 4 = REM
CLASSES_STR = [LABEL_TO_NAME.get(int(label), str(label)) for label in CLASSES]

y_pred = rf_model.predict(X_cv)
y_probabilities = rf_model.predict_proba(X_cv)

pred_df = pd.DataFrame(
    {
        "true_label": y_cv.values,
        "pred_label": y_pred,
    }
)

pred_df["true_label_string"] = pred_df["true_label"].map(LABEL_TO_NAME)
pred_df["pred_label_string"] = pred_df["pred_label"].map(LABEL_TO_NAME)
pred_df["correct"] = (pred_df["true_label"] == pred_df["pred_label"]).astype(int)

probability_name = [f"p_{LABEL_TO_NAME.get(int(label), str(label))}" for label in CLASSES]
pred_df[probability_name] = y_probabilities

pred_df = pd.concat([cv_meta, pred_df], axis=1)

pred_df = apply_mode_filter(pred_df, 5)

pred_df["pred_label_mode_filter_string"] = pred_df["pred_label_mode_filter"].map(LABEL_TO_NAME)

y_pred_mode_filter = pred_df["pred_label_mode_filter"].values

pred_df.to_csv(OUTPUT_DIRECTORY / f"predictions_{MODEL_VERSION}.csv", index=False)


# Calculate overall metrics
overall_metrics = {
    "accuracy": float(accuracy_score(y_cv, y_pred_mode_filter)),  # Overall accuracy
    "macro_f1": float(f1_score(y_cv, y_pred_mode_filter, average="macro")),  # Macro F1 average
    "weighted_f1": float(f1_score(y_cv, y_pred_mode_filter, average="weighted")),  # Weighted F1 average
    "cohen_kappa": float(cohen_kappa_score(y_cv, y_pred_mode_filter)),  # Cohen's kappa
    "macro_auc": float(roc_auc_score(y_cv, y_probabilities, multi_class="ovr", average="macro")),
    "weighted_auc": float(
        roc_auc_score(y_cv, y_probabilities, multi_class="ovr", average="weighted")
    ),
    "log_loss": float(log_loss(y_cv, y_probabilities)),
}

importances = rf_model.feature_importances_

feature_importance = {
    feature_columns[i]: float(importances[i]) for i in range(len(feature_columns))
}

feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

per_class = classification_report(
    y_cv, y_pred_mode_filter, labels=CLASSES, output_dict=True
)

per_class_auc = {}
plt.figure(figsize=(10, 8))
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title(f"One-vs-Rest ROC Curves: {MODEL_VERSION}")


for i, label in enumerate(CLASSES):
    y_binary = (y_cv == label).astype(int)
    y_probability_current_class = y_probabilities[:, i]
    fpr, tpr, roc_thresholds = roc_curve(y_binary, y_probability_current_class)
    class_auc = auc(fpr, tpr)
    per_class_auc[CLASSES_STR[i]] = class_auc
    plt.plot(fpr, tpr, label=f"ROC curve for class {CLASSES_STR[i]} (AUC = {class_auc:.2f})")

plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_DIRECTORY / f"roc_curves_{MODEL_VERSION}.png")


fig, ax = plt.subplots(figsize=(10, 8))
confusion = confusion_matrix(y_cv, y_pred_mode_filter, labels=CLASSES)  # Confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=CLASSES_STR)
disp.plot(ax=ax, cmap="Blues")


ax.set_title(f"Confusion Matrix: {MODEL_VERSION}")
plt.tight_layout()

plt.savefig(OUTPUT_DIRECTORY / f"confusion_matrix_{MODEL_VERSION}.png")

rf_parameters_payload = {
    "Version": MODEL_VERSION,
    "Notes": "Hyperparamter tuning",
    "rf_parameters": rf_parameters,
}

metrics_payload = {
    "overall": overall_metrics,
    "per_class": per_class,
    "per_class_auc": per_class_auc,
    "feature_importance": feature_importance,
}

with open(OUTPUT_DIRECTORY / f"metrics_{MODEL_VERSION}.json", "w") as f:
    json.dump(metrics_payload, f, indent = 2, sort_keys = True)

with open(OUTPUT_DIRECTORY / f"params_{MODEL_VERSION}.json", "w") as f:
    json.dump(rf_parameters_payload, f, indent = 2, sort_keys = True)
