import pandas as pd

def count_frequencies(word_data, label_data):
   
    # stworzenie słownika zawierającego częstotliwości występowania słów w wiadomościach porządanych i nieporządanych
    word_freq_dict = {}
    # stworzenie słownika zawierającego liczbę wszystkich słów dla obu kategorii
    words_per_class_dict = {
        'ham': 0,
        'spam': 0
    }
    
    # iteracja po wiadomościahc w zbiorze danych, następnie po słowach w wiadomości i zliczanie ich w przygotowanych słownikach według etykiety wiadomości
    for index, row in word_data.iterrows():
        label = label_data.iloc[index]
        for word in row['words']:
            if word not in word_freq_dict.keys():
                word_freq_dict[word] = {"spam": 0, "ham": 0}
            
            if label == 'spam':
                word_freq_dict[word]["spam"] += 1
                words_per_class_dict['spam'] += 1
            else:
                word_freq_dict[word]["ham"] += 1
                words_per_class_dict['ham'] += 1

    return (word_freq_dict, words_per_class_dict)

def calculate_word_probabilities(word_freq_dict, words_per_class_dict, vocab_size, alpha = 1):
    
    # stworzenie słownika do przechowania prawdopodobieństw wystąpienia każdego słowa w wiadomości porządanej oraz nieporządanej
    word_prob_dict = {}
    
    # uzupełnianie słownika prawdopodobieństwami zgodnie z algorytmem wykorzystującym wygładzanie Laplace'a
    for word, freq in word_freq_dict.items():
        word_prob_dict[word] = {}
        
        word_prob_dict[word]['spam'] = (freq['spam'] + alpha) / (words_per_class_dict['spam'] + alpha*vocab_size)
        
        word_prob_dict[word]['ham'] = (freq['ham'] + alpha) / (words_per_class_dict['ham'] + alpha*vocab_size)
    
    return word_prob_dict

# Wpólne wywołanie powyższych funkcji
# Funkcja zwracająca wypełniony słownik prawdopodobieństw słów
def make_word_probability_dict(training_word_dataset, training_label_dataset):
    
    word_freq_dict, words_per_class_dict = count_frequencies(training_word_dataset, training_label_dataset)
    vocab_size = len(word_freq_dict)
    word_prob_dict = calculate_word_probabilities(word_freq_dict, words_per_class_dict, vocab_size)

    return word_prob_dict