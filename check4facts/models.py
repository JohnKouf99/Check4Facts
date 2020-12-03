from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from joblib import dump, load


class SklearnBaseModel:

    def __init__(self, model):
        self.model = model

    def fit(self, x, y):
        # TODO uncomment below when enough train instances are in db
        # scores = cross_val_score(self.model, x, y, cv=5)
        # print("Acc: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        self.model = self.model.fit(x, y)
        return self.model

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def save(self, path):
        dump(self.model, path)

    def load(self, path):
        self.model = load(path)
        return self


class LR(SklearnBaseModel):
    def __init__(self, **kwargs):
        super().__init__(LogisticRegression(**kwargs))


class RF(SklearnBaseModel):
    def __init__(self, **kwargs):
        super().__init__(RandomForestClassifier(**kwargs))
