# src/stage_3/train_eval.py

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
import numpy as np
from joblib import dump
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


MODEL_VERSION = "rf_model_cv_v7"
OUTPUT_DIRECTORY = Path("models") / MODEL_VERSION
OUTPUT_DIRECTORY.mkdir(parents = True, exist_ok = True)
WAKE_LABEL = 0
N1_LABEL = 1
REM_LABEL = 4

X_train, y_train, X_cv, y_cv, X_test, y_test, feature_columns, cv_meta = generate_training_splits()

count = np.bincount(y_train)
target = int(0.60 * count[WAKE_LABEL])
n1_count = count[1]
rem_count = count[4]


sampling_strategy = {}
for label in [1]:
    sampling_strategy[label] = target


# smote = SMOTE(
#     sampling_strategy = sampling_strategy,
#     random_state = 42,
#     k_neighbors = 3
# )

smote = BorderlineSMOTE(
    sampling_strategy = sampling_strategy,
    k_neighbors = 3,
    random_state = 42,
)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# rf_base = RandomForestClassifier(
#     n_jobs = -1,
#     random_state = 42,
# )

# parameter_distributions = {
#     "n_estimators": [300, 400, 500, 600, 700, 800, 900, 1000],
#     "max_depth": [None, 10, 12, 16, 20, 24, 28, 30],
#     "min_samples_leaf": [1, 2, 4, 8],
#     "min_samples_split": [2, 4, 6, 10],
#     "max_features": ["sqrt", "log2"],
#     "class_weight": ["balanced", None],
#     "bootstrap": [True, False],
#     "criterion": ["gini", "entropy"],
# }

# rf_search = RandomizedSearchCV(
#     estimator = rf_base,
#     param_distributions = parameter_distributions,
#     n_iter = 25,
#     scoring = "f1_macro",
#     cv = 3,
#     verbose = 3,
#     random_state = 42,
#     n_jobs = -1,
# )

# rf_search.fit(X_train_resampled, y_train_resampled)

# best_params = rf_search.best_params_

# rf_model = RandomForestClassifier(**best_params, n_jobs = -1, random_state = 42)
# rf_model.fit(X_train_resampled, y_train_resampled)


rf_parameters = {
    "bootstrap": False,
    "class_weight": "balanced_subsample",
    "criterion": "gini",
    "max_depth": None,
    "max_features": "log2",
    "min_samples_leaf": 1,
    "n_jobs": -1,
    "min_samples_split": 2,
    "n_estimators": 500,
    "random_state": 42
}

rf_model = RandomForestClassifier(**rf_parameters)

rf_model.fit(X_train_resampled, y_train_resampled)

dump(rf_model, OUTPUT_DIRECTORY / f"{MODEL_VERSION}.joblib")

y_pred = rf_model.predict(X_cv)
y_probabilities = rf_model.predict_proba(X_cv)

# Calculate overall metrics
overall_metrics = {
    "accuracy": float(accuracy_score(y_cv, y_pred)),  # Overall accuracy
    "macro_f1": float(f1_score(y_cv, y_pred, average="macro")),  # Macro F1 average
    "weighted_f1": float(f1_score(y_cv, y_pred, average="weighted")),  # Weighted F1 average
    "cohen_kappa": float(cohen_kappa_score(y_cv, y_pred)),  # Cohen's kappa
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

CLASSES = list(rf_model.classes_)  # 0 = W, 1 = N1, 2 = N2, 3 = N3, 4 = REM
LABEL_TO_NAME = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
CLASSES_STR = [LABEL_TO_NAME.get(int(label), str(label)) for label in CLASSES]

per_class = classification_report(
    y_cv, y_pred, labels=CLASSES, output_dict=True
)  # Calculates per class precision, recall, F1, support

per_class_auc = {}
plt.figure(figsize=(10, 8))
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title(f"One-vs-Rest ROC Curves: {MODEL_VERSION}")


for i, label in enumerate(CLASSES):
    y_binary = (y_cv == label).astype(int)
    y_probability_current_class = y_probabilities[:, i]
    fpr, tpr, thresholds = roc_curve(y_binary, y_probability_current_class)
    class_auc = auc(fpr, tpr)
    per_class_auc[CLASSES_STR[i]] = class_auc
    plt.plot(fpr, tpr, label=f"ROC curve for class {CLASSES_STR[i]} (AUC = {class_auc:.2f})")

plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_DIRECTORY / f"roc_curves_{MODEL_VERSION}.png")


fig, ax = plt.subplots(figsize=(10, 8))
confusion = confusion_matrix(y_cv, y_pred, labels=CLASSES)  # Confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=CLASSES_STR)
disp.plot(ax=ax, cmap="Blues")


ax.set_title(f"Confusion Matrix: {MODEL_VERSION}")
plt.tight_layout()

plt.savefig(OUTPUT_DIRECTORY / f"confusion_matrix_{MODEL_VERSION}.png")

rf_parameters_payload = {
    "Notes": "v7: v3 SMOTE applied to N1, but no SMOTE to REM. v6 Hyperparameters applied.",
    "rf_parameters": rf_parameters,

}

with open(OUTPUT_DIRECTORY / f"params_{MODEL_VERSION}.json", "w") as f:
    json.dump(rf_parameters_payload, f, indent = 2, sort_keys = True)


metrics_payload = {
    "overall": overall_metrics,
    "per_class": per_class,
    "per_class_auc": per_class_auc,
    "feature_importance": feature_importance,
}

with open(OUTPUT_DIRECTORY / f"metrics_{MODEL_VERSION}.json", "w") as f:
    json.dump(metrics_payload, f, indent = 2, sort_keys = True)

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

pred_df.to_csv(OUTPUT_DIRECTORY / f"predictions_{MODEL_VERSION}.csv", index=False)
