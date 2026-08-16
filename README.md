# ML Classification Assignment

## Problem Statement

Build a classification system to predict breast cancer diagnosis
(Benign vs Malignant) from cell nucleus measurements. Implement six machine
learning models, evaluate them using standard metrics, and deploy an interactive
Streamlit web application on Streamlit Community Cloud.

## Dataset Description

Breast cancer is the most common cancer amongst women in the world. It accounts
for 25% of all cancer cases, and affected over 2.1 Million people in 2015 alone.
It starts when cells in the breast begin to grow out of control. These cells
usually form tumors that can be seen via X-ray or felt as lumps in the breast
area.

The key challenges against it’s detection is how to classify tumors into
malignant (cancerous) or benign(non cancerous). We ask you to complete the
analysis of classifying these tumors using machine learning (with SVMs) and the
Breast Cancer Wisconsin (Diagnostic) Dataset.

## Github Repository Link

https://github.com/2025ab05234-kc/assignment-2.git

## Models Used

### Comparison Table — Evaluation Metrics

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression      | 0.9649   | 0.9970 | 0.9318    | 0.9762 | 0.9535 | 0.9260 |
| Decision Tree            | 0.9211   | 0.9484 | 0.9024    | 0.8810 | 0.8916 | 0.8297 |
| kNN                      | 0.9561   | 0.9835 | 0.9744    | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes              | 0.9825   | 0.9974 | 0.9762    | 0.9762 | 0.9762 | 0.9623 |
| Random Forest (Ensemble) | 0.9649   | 0.9980 | 0.9750    | 0.9286 | 0.9512 | 0.9245 |

### Observations on Model Performance

| ML Model Name            | Observation about model performance                                                                                                                                   |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | High recall (97.62%) with strong AUC (0.9970). Slightly lower precision (0.93). Simple, interpretable, and fast.                                                      |
| Decision Tree            | Lowest accuracy (92.11%) among models. AUC (0.9484) with entropy criterion. Single tree more interpretable but prone to variance.                                     |
| kNN                      | Good performance (95.61% accuracy). Balanced metrics. AUC (0.9835) shows good discrimination. Sensitive to feature scaling; benefits from StandardScaler in pipeline. |
| Naive Bayes              | Tied for best accuracy (98.25%). Excellent balance — precision and recall both 97.62%. High AUC (0.9974). Strong improvement with tuned threshold.                    |
| Random Forest (Ensemble) | Highest AUC (0.9980). Good precision (0.975) and recall (0.929). Ensemble reduces overfitting; robust and generalizes well.                                           |

| Overall Winner for your dataset? | **Naive Bayes**. Tied for the best accuracy (98.25%) with excellent balance (Precision and Recall both at 97.62%) and a high AUC (0.9974). Performs exceptionally well on this dataset. |
