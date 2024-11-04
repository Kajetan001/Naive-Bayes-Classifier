from data_prep import text_into_list
import pandas as pd

def make_class_count_dict(training_label_dataset):
    
    # tworzenie słownika z ilościami wiadomości określonych jako spam oraz ham
    class_count_dict = {
        'ham': int(training_label_dataset.value_counts().iloc[0]),
        'spam': int(training_label_dataset.value_counts().iloc[1])
    }
    return class_count_dict

def calculate_class_score(class_count_dict, words, word_prob_dict):
    # liczba wszystkich wiadomości w zbiorze danych
    message_count = class_count_dict["spam"] + class_count_dict["ham"]

    # liczenie ogólnych prawdopodobieństw na wystąpnienie wiadomości porządanej i nieporządanej
    spam_prob = class_count_dict["spam"] / message_count
    ham_prob = class_count_dict["ham"] / message_count

    # wyliczanie punktacji danej wiadomości poprzez mnożenie ogólnego prawdopodobieństwa na wystąpienie wiadomości z danej kategorii
    # przez prawdopodobieństwa każdego ze słów na znalezienie się w tej kategorii
    for word in words:
        if word in word_prob_dict.keys():
            spam_prob *= word_prob_dict[word]['spam']
            ham_prob *= word_prob_dict[word]['ham']
            
    return (spam_prob, ham_prob)

def classify(text, training_label_dataset, word_prob_dict):
    
    # czyszczenie tekstu, który poddamy klasyfikacji
    words = text_into_list(text)

    # stworzenie słownika z ilościami wiadomości określonych jako spam oraz ham
    class_count_dict = make_class_count_dict(training_label_dataset)
    spam_prob, ham_prob = calculate_class_score(class_count_dict, words, word_prob_dict)

    # porównanie punktacji; wyższa wartość punktacji spamu oznacza klasyfikację wiadomości jako spam
    if spam_prob > ham_prob:
        return 'spam'
    else:
        return 'ham'

def make_prediction(training_label_dataset, word_prob_dict, test_word_dataset):
    
    # funkcja lambda aplikująca funkcję calssify
    f = lambda x : classify(x, training_label_dataset, word_prob_dict)
    # zastosowanie funkcji lambda na zbiorze testowym
    y_pred = test_word_dataset['messages'].apply(f)

    return y_pred