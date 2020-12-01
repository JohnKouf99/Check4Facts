from sklearn.metrics import accuracy_score, f1_score


def accuracy(y_true, y_pred):
    score = accuracy_score(y_true, y_pred)
    return score


def f1(y_true, y_pred):
    score = f1_score(y_true, y_pred)
    return score
