import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import e

eps = 0.000001


def sigmoid(x):
    # x = x-x.max()
    z = 1 / (1 + np.exp(-(x + eps)))
    return z


class LogisticRegression:
    def __init__(self):
        self.weights_ = None
        self.b_ = 0

    def init_weights(self, X):
        self.weights_ = np.random.rand(X.shape[1], 1) / 1000
        self.b_ = 0

    def predict(self, X):
        z = self.hypothesis(X)
        predictions = z >= 0.5
        predictions = np.array(predictions.astype(int))
        return predictions

    def predict_proba(self, X: pd.DataFrame):
        return self.hypothesis(X)

    def hypothesis(self, X):
        hypo = X @ self.weights_ + self.b_
        hypo = sigmoid(hypo)
        return hypo

    def cross_entropy(self, y: np.array, predictions: np.array):
        return np.mean(-1 * (y * np.log(predictions + eps) + (1 - y) * np.log(1 - predictions + eps)))

    def compute_grades(self, X, y_hat, y):
        cost = self.cross_entropy(y, y_hat)
        dw = ((y_hat - y).T @ X) / X.shape[0]
        dw = dw.T
        db = (y_hat - y).mean()
        return cost, dw, db

    def update_weights(self, alpha, dw, db):
        self.weights_ = self.weights_ - alpha * dw
        self.b_ = self.b_ - alpha * db

    def fit(self, X, y, n_iterations=2000, alpha=0.01, thresh=0.1):
        y = np.array(y)
        y = y[:, np.newaxis]
        self.init_weights(X)

        for i in range(n_iterations):
            y_hat = self.hypothesis(X)
            loss, dw, db = self.compute_grades(X, y_hat, y)
            self.update_weights(alpha, dw, db)
            # Calculate the norm of the gradients
            gradient_norm = np.linalg.norm(dw) + np.abs(db)
            gradient_norm = gradient_norm.sum()
            # Exit the loop if the gradient norm is below the threshold
            if gradient_norm < thresh:
                print(f"Stopping early at iteration {i} with gradient norm {gradient_norm}")
                break

        return self.weights_

    def score(self, predictions, actual):
        return (predictions == actual).mean()

    def compute_roc_curve(self, y_true, y_prob):
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        # Sort instances by their predicted probabilities in descending order
        sorted_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[sorted_indices]
        y_prob_sorted = y_prob[sorted_indices]
        # Initialize variables
        tpr = []
        fpr = []
        thresholds = []
        tp = 0
        fp = 0
        fn = np.sum(y_true == 1)
        tn = np.sum(y_true == 0)
        # Iterate through the sorted list and compute TPR and FPR
        for i in range(len(y_true)):
            if y_true_sorted[i] == 1:
                tp += 1
                fn -= 1
            else:
                fp += 1
                tn -= 1

            tpr.append(tp / (tp + fn))
            fpr.append(fp / (fp + tn))
            thresholds.append(y_prob_sorted[i])
        return np.array(fpr), np.array(tpr), np.array(thresholds)

    def plot_roc_curve(self, X, y):
        # Predict probabilities for the positive class
        y_prob = self.hypothesis(X).flatten()
        # Compute ROC curve
        fpr, tpr, thresholds = self.compute_roc_curve(y, y_prob)
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.show()
        return self.find_optimal_threshold(fpr, tpr, thresholds)

    def find_optimal_threshold(self, fpr, tpr, thresholds):
        # Calculate the distance to the top-left corner (0,1)
        distances = np.sqrt((fpr - 0) ** 2 + (tpr - 1) ** 2)
        # Find the index of the minimum distance
        optimal_idx = np.argmin(distances)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold
