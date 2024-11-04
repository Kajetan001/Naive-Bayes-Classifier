from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Funkcja dokonująca klasyfikacji przy pomocy metod z biblioteki sklearn
def get_sklearn_nb_pred(X_train, X_test, y_train):
    
    # inicjalizacja vektoryzatora
    vectorizer = CountVectorizer()

    # dopasowanie i transformacja danych treningowych przy pomocy wektoryzatora
    X_train_counts = vectorizer.fit_transform(X_train["words"])

    # transformacja danych testowych przy pomocy wektoryzatora
    X_test_counts = vectorizer.transform(X_test["words"])

    # trening modelu
    model = MultinomialNB()
    model.fit(X_train_counts, y_train)

    # przeprowadzenie klasyfikacji
    y_pred = model.predict(X_test_counts)

    return y_pred