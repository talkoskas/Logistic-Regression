from sklearn import datasets
from MultinomialLogisticRegression import MultinomialLR
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def main():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.get('data'))
    y = np.array(iris.get('target'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, )
    model = MultinomialLR()
    model.fit(X_train, y_train, n_iterations=5000)
    print("Probabilities of the model over the test data set\n", model.predict_proba(X_test))
    print("Predictions of the model over the test data set:\n", model.predict(X_test))
    print("\nScore of the model over the test data set: ", model.score(X_test, y_test))



if __name__ == "__main__":
    main()