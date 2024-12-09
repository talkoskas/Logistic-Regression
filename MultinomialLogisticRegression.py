import numpy as np
import pandas as pd
from LogisticRegression import LogisticRegression


class MultinomialLR:
    def __init__(self):
        self.models = {}

    def fit(self, X, y, n_iterations=2000, alpha=0.01, thresh=0.1):
        unique_labels = np.unique(np.array(y))
        models = {}
        for label in unique_labels:
            curr_labels = np.where(y == label, 1, 0)
            model = LogisticRegression()
            model.fit(X, curr_labels, n_iterations=n_iterations, alpha=alpha, thresh=thresh)
            models[label] = model
        self.models = models



    def predict_proba(self, X):
        predictions = {}
        labels = self.models.keys()
        for label in labels:
            model = self.models[label]
            tmp = model.predict_proba(X).iloc[:, 0]
            predictions[label] = pd.Series(tmp, name=label)

        predictions_df = pd.DataFrame(predictions)
        return predictions_df

    def predict(self, X):
        predictions_df = self.predict_proba(X)

        # Get the index (label) of the max probability for each instance
        predicted_labels = predictions_df.idxmax(axis=1)

        return predicted_labels.to_numpy()

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = (predictions == y).mean()
        return accuracy
