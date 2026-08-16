from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.naive_bayes import GaussianNB


def make_naive_bayes():
    return Pipeline(
        [
            ("transform", PowerTransformer(method="yeo-johnson")),
            ("clf", GaussianNB(var_smoothing=1e-9)),
        ]
    )
