from data_prep import *
from probabilities import *
from naive_bayes_classifier import *

import pandas as pd
import pytest

"""
Plik testowy modułu pytest - unit testy
Przeprowadzamy testy, wywołując funkcje naszego algorytmu na niedużym zbiorze wiadomości, dla których wszystkie liczności i wartości prawdopodobieństw
policzyliśmy ręcznie. Następnie porównujemy wyniki funkcji z oczekiwanymi wartościami przy pomocy słowa kluczowego assert.
Sprawdzamy funkcje count_frequencies, calculate_word_probabilities oraz calculate_class_score.
"""

def sample_train_data(): 
    train_data = pd.DataFrame({
        "label": ["ham", "ham", "spam"],
        "messages": [
            "Are we still on for lunch tomorrow?",
            "Hi, can you please send me the final report by tomorrow?",
            "Congratulations! You have won gift card. Please claim money reward."
        ]
    })
    
    train_data['words'] = train_data['messages'].apply(text_into_list)
    word_train, label_train = train_data.iloc[:,1:], train_data.iloc[:,0]

    return (word_train, label_train)

def test_count_frequencies():
    word_train, label_train = sample_train_data()

    word_freq_dict, words_per_class_dict = count_frequencies(word_train, label_train)

    assert word_freq_dict["are"]["ham"] == 1
    assert word_freq_dict["are"]["spam"] == 0
    assert word_freq_dict["you"]["ham"] == 1
    assert word_freq_dict["you"]["spam"] == 1
    assert word_freq_dict["tomorrow"]["ham"] == 2
    assert word_freq_dict["tomorrow"]["spam"] == 0
    assert word_freq_dict["congratulations"]["ham"] == 0
    assert word_freq_dict["congratulations"]["spam"] == 1
    assert word_freq_dict["claim"]["ham"] == 0
    assert word_freq_dict["claim"]["spam"] == 1

    assert words_per_class_dict["ham"] == 18
    assert words_per_class_dict["spam"] == 10

def test_calculate_word_probabilities():
    word_train, label_train = sample_train_data()

    word_freq_dict, words_per_class_dict = count_frequencies(word_train, label_train)
    vocab_size = len(word_freq_dict)
    word_prob_dict = calculate_word_probabilities(word_freq_dict, words_per_class_dict, vocab_size)

    # (freq['spam'] + alpha) / (words_per_class_dict['spam'] + alpha*vocab_size)
    assert word_prob_dict["are"]["ham"] == (1 + 1) / (18 + (1 * 25))
    assert word_prob_dict["are"]["spam"] == (0 + 1) / (10 + (1 * 25))
    assert word_prob_dict["you"]["ham"] == (1 + 1) / (18 + (1 * 25))
    assert word_prob_dict["you"]["spam"] == (1 + 1) / (10 + (1 * 25))
    assert word_prob_dict["tomorrow"]["ham"] == (2 + 1) / (18 + (1 * 25))
    assert word_prob_dict["tomorrow"]["spam"] == (0 + 1) / (10 + (1 * 25))
    assert word_prob_dict["congratulations"]["ham"] == (0 + 1) / (18 + (1 * 25))
    assert word_prob_dict["congratulations"]["spam"] == (1 + 1) / (10 + (1 * 25))
    assert word_prob_dict["claim"]["ham"] == (0 + 1) / (18 + (1 * 25))
    assert word_prob_dict["claim"]["spam"] == (1 + 1) / (10 + (1 * 25))

def sample_test_data():
    test_data = pd.DataFrame({
        "label": ["ham", "spam"],
        "messages": [
            "Hi, are you doing ok?",
            "Please send money to get reward"
        ]
    })

    test_data['words'] = test_data['messages'].apply(text_into_list)
    word_test = test_data.iloc[:,1:]

    return word_test

def test_calculate_class_score():
    word_train, label_train = sample_train_data()

    word_freq_dict, words_per_class_dict = count_frequencies(word_train, label_train)
    vocab_size = len(word_freq_dict)
    word_prob_dict = calculate_word_probabilities(word_freq_dict, words_per_class_dict, vocab_size)

    word_test = sample_test_data()

    class_count_dict = make_class_count_dict(label_train)
    
    spam_prob_0, ham_prob_0 = calculate_class_score(class_count_dict, word_test['words'][0], word_prob_dict)
    spam_prob_1, ham_prob_1 = calculate_class_score(class_count_dict, word_test['words'][1], word_prob_dict)

    manual_spam_prob_0 = ((1 / 3) * 
                           ((0 + 1) / (10 + (1 * 25))) * # hi
                           ((0 + 1) / (10 + (1 * 25))) * # are
                           ((1 + 1) / (10 + (1 * 25)))) # you
    
    manual_ham_prob_0 = ((2 / 3) * 
                           ((1 + 1) / (18 + (1 * 25))) * # hi
                           ((1 + 1) / (18 + (1 * 25))) * # are
                           ((1 + 1) / (18 + (1 * 25)))) # you

    
    manual_spam_prob_1 = ((1 / 3) * 
                           ((1 + 1) / (10 + (1 * 25))) * # please
                           ((0 + 1) / (10 + (1 * 25))) * # send
                           ((1 + 1) / (10 + (1 * 25))) * # money
                           ((1 + 1) / (10 + (1 * 25)))) # reward
    
    manual_ham_prob_1 = ((2 / 3) * 
                           ((1 + 1) / (18 + (1 * 25))) * # please
                           ((1 + 1) / (18 + (1 * 25))) * # send
                           ((0 + 1) / (18 + (1 * 25))) * # money
                           ((0 + 1) / (18 + (1 * 25)))) # reward
    
    assert spam_prob_0 == manual_spam_prob_0
    assert ham_prob_0 == manual_ham_prob_0
    assert spam_prob_1 == manual_spam_prob_1
    assert ham_prob_1 == manual_ham_prob_1