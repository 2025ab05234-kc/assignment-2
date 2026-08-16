import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


def compute_binary_metrics(y_true, y_pred, y_proba):
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(auc),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def compute_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, digits=4, zero_division=0)
