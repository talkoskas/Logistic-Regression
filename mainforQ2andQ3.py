import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from LogisticRegression import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # Step 1: Load the data
    df = pd.read_csv('spam_ham_dataset.csv')
    df.columns = ['Unnamed Integer', 'Label String', 'Mail Content', 'Label Number']
    df = pd.DataFrame(df)
    y = df['Label Number']

    df.drop(columns=['Label String', 'Label Number'], inplace=True)



    # Extract the column to be vectorized
    mail_content = df['Mail Content']
    # Vectorize the mail content
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(mail_content)

    # Convert the TF-IDF matrix to a DataFrame
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

    # Concatenate the new TF-IDF DataFrame with the original DataFrame (excluding 'Mail Content')
    df_updated = pd.concat([df.drop(columns=['Mail Content']), X_tfidf_df], axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_updated)
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize logistic regression model
    model = LogisticRegression()
    weights = model.fit(X_train, y_train, n_iterations=10000, alpha=0.33).flatten()
    print("The weights vector after fit:\n", weights)
    predicted_proba = model.predict_proba(X_test).flatten()
    print("\nProbabilities predicted for the test data set-\n", predicted_proba)
    # Predict and evaluate the model
    y_pred = model.predict(X_test).flatten()
    print("\nLabels predicted for the test data set:\n", y_pred)
    accuracy = model.score(y_test, y_pred)
    print("\nAccuracy of the model on test data set: ", accuracy)

    # Question 3 answer below
    threshold = model.plot_roc_curve(X_test, y_test)
    print("\n\nOptimal threshold found at ", threshold)
    print("\nExplanation: this is the point that has the most True Positive rate,")
    print("and has the least False Positive rate in order to minimize errors")

if __name__ == "__main__":
    main()
