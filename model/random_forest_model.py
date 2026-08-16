from sklearn.ensemble import RandomForestClassifier


def make_random_forest(random_state: int = 42):
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
