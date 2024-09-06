import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Load the dataset using pandas
def load_data():
    # Load CSV file
    df = pd.read_csv(<my csv path>, encoding='utf-8')
    
    # Inspect the first few rows to verify data
    print(df.head())
    
    # Use 'Overview' as the text data and 'IMDB_Rating' as the target
    data = df['Overview']  # Text data
    target = df['IMDB_Rating']  # Target variable

    # For classification, convert IMDB_Rating into categories (e.g., high/low)
    # Here we create a simple binary classification based on a threshold rating
    threshold = 7.0  # Example threshold
    target = (target >= threshold).astype(int)  # 1 if rating >= threshold, 0 otherwise
    
    return data, target

# Preprocess and split the data
def preprocess_data(data, target):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

# Train the model
def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(metrics.classification_report(y_test, y_pred))

# Main function
def main():
    data, target = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data, target)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
