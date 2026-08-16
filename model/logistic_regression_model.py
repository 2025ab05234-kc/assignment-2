from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def make_logistic_regression(random_state: int = 42):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=10)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=random_state,
                    class_weight="balanced",
                    C=0.5,
                ),
            ),
        ]
    )
