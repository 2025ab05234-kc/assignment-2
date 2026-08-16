from sklearn.tree import DecisionTreeClassifier


def make_decision_tree(random_state: int = 42):
    return DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced",
        criterion="gini",
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        ccp_alpha=0.005,
    )
