def sparsity_scorer(
    clf: LogisticRegression | Pipeline, X=None, y=None, scale_coef=1.0
) -> float:
    """
    Evaluates the sparsity of the estimator within a pipeline
    Can work with a LogisticRegression or a pipeline containing
    a LogisticRegression

    Parameters:
    -----------
        clf: a LogisticRegression or a Pipeline
        X: ndarray (unused)
        y: ndarray (unused)

    Returns
    -------
        sparsity: float in [0., 1.]
                  0. means all the dimensions are used,
                  1. means no dimension are used
    """
    if isinstance(clf, Pipeline):
        clf_item = None
        coef = scale_coef
        for k, v in clf.named_steps.items():
            if isinstance(v, SelectorMixin):
                coef *= v.get_support().sum() / v.get_support().size
            elif isinstance(v, LogisticRegression):
                clf_item = v
        if clf_item is None:
            raise RuntimeError("Cannot estimate the sparsity of the pipeline")
        else:
            return sparsity_scorer(clf_item, X, y, coef)
    elif isinstance(clf, LogisticRegression):
        non_null_coefs = clf.coef_.ravel() != 0
        sparsity = 1.0 - non_null_coefs.sum() / non_null_coefs.size * scale_coef
        return sparsity
    else:
        raise RuntimeError(f"Cannot estimate the sparsity of a {type(clf)}")
