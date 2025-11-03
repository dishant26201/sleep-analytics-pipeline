from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
)

from .load_data import generate_training_splits  


# OVERALL METRICS:
# Accuracy
# Macro F1
# Weighted F1
# Cohen’s Kappa
# ROC–AUC Curve

# PER CLASS METRICS
# Precision
# Recall
# F1-score
# Support
# ROC–AUC Curves

# VISUALS:
# Confusion Matrix
# Feature Importance

# OUTPUTS
# metrics_rf_v#.json
# confusion_matrix_rf_v#.png
# roc_curves_rf_metric_name_v#.png
# roc_curves_rf_overall_v#.png




X_train, y_train, X_cv, y_cv, X_test, y_test, feature_columns = generate_training_splits()

unique_classes = np.unique(y_train)
weights = compute_class_weight(class_weight = "balanced", classes = unique_classes, y = y_train)
class_weights = dict(zip(unique_classes, weights))

CLASSES = [0, 1, 2, 3, 4] # 0 = W, 1 = N1, 2 = N2, 3 = N3, 4 = REM
CLASSES_STR = ["W", "N1", "N2", "N3", "REM"]


MODEL_VERSION = "rf_v1"
OUTPUT_DIRECTORY = Path("models") / MODEL_VERSION
OUTPUT_DIRECTORY.mkdir(parents = True, exist_ok=True)

# Minimal RF parameters 
RF_PARAMETERS = dict(
    n_estimators = 500,
    min_samples_leaf = 1,
    max_depth = None,
    class_weight = class_weights,
    n_jobs = 1,
    random_state = 42
)

rf_model = RandomForestClassifier(**RF_PARAMETERS)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_cv)
y_probabilities = rf_model.predict_proba(X_cv)
rf_model_parameters = rf_model.get_params(deep = True)

# Overall metrics

overall = {
    "accuracy": float(accuracy_score(y_cv, y_pred)),
    "macro_f1": float(f1_score(y_cv, y_pred, average = "macro")),
    "weighted_f1": float(f1_score(y_cv, y_pred, average = "weighted")),
    "cohen_kappa": float(cohen_kappa_score(y_cv, y_pred)),
}

per_class = classification_report(y_cv, y_pred, labels = CLASSES, output_dict = True)

confusion = confusion_matrix(y_cv, y_pred, labels = CLASSES)
df = pd.DataFrame(confusion, index = CLASSES_STR, columns = CLASSES_STR)

plt.figure(figsize = (10, 7)) 
sns_plot = sns.heatmap(df, annot = True)

