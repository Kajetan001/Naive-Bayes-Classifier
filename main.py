from data_prep import *
from probabilities import *
from naive_bayes_classifier import *
from sklearn_naive_bayes import *

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

"""
Do klasyfikacji przy użyciu naiwnego klasyfikatora Bayesa zdecydowaliśmy się użyć zbiorów danych zawierającego wiadomości
SMS Spam Collection: https://archive.ics.uci.edu/dataset/228/sms+spam+collection oraz Telegram Spam Ham: https://huggingface.co/datasets/thehamkercat/telegram-spam-ham ,
oznaczone jako porządane lub nieporządane.
Do wyuczenia klasyfikatora skorzystaliśmy z większego zbioru Telegram, natomiast później użyjemy zbioru SMS, sprawdzając na nim dokładność algorytmu.
Filtry spamu są klasycznym przykładem użycia klasyfikatora Bayesa. Korzystamy z bibliotek pandas i scikit-learn do stworzenia klasyfikatora.
"""

if __name__ == "__main__" :

    # załadowanie zbioru danych Telegram
    column_names = ['label', 'messages']
    data = load_dataset('message_data/TelegramSpamHam.csv', ',', column_names)
    # czyszczenie tekstu, podział na dane i etykiety testowe i treningowe
    X_train, X_test, y_train, y_test = prepare_data(data, text_into_list, column_names)

    # tworzenie słownika z prawdopodobieństwami wystąpienia każdego słowa w wiadomości porządanej oraz nieporządanej
    word_prob_dict = make_word_probability_dict(X_train, y_train)

    # przeprowadzenie klasyfikacji
    y_pred = make_prediction(y_train, word_prob_dict, X_test)

    # drukowanie raportu klasyfikacji
    print('Raport dla naiwnego klasyfikatora Bayesa "from scratch", na danych ze zbioru Telegram')
    print(classification_report(y_test, y_pred))

    # przygotowanie danych do klasyfikatora z biblioteki sklearn
    # wyczyszczony tekst przechowujemy jako string zamiast jako listę
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = prepare_data(data, text_into_string, column_names)
    y_pred_sk = get_sklearn_nb_pred(X_train_sk, X_test_sk, y_train_sk)

    # drukowanie raportu klasyfikacji dla algorytmu sklearn
    print('Raport dla naiwnego klasyfikatora Bayesa z biblioteki sklearn, na danych ze zbioru Telegram')
    print(classification_report(y_test_sk, y_pred_sk))
    
    # załadowanie zbioru danych SMS
    sms_data = load_dataset("message_data/SMSSpamCollection", '\t', column_names)
    # podział na kolumnę etykiet oraz dane tekstowe
    X_sms, y_sms = sms_data.iloc[:,1:], sms_data.iloc[:,0]
    
    # przeprowadzenie klasyfikacji szkolonym wcześniej klasyfikatorem
    y_pred_sms = make_prediction(y_train, word_prob_dict, X_sms)

    # drukowanie raportu klasyfikacji
    print('Raport dla naiwnego klasyfikatora Bayesa "from scratch", na danych testowych ze zbioru SMS')
    print(classification_report(y_sms, y_pred_sms))

    # przygotowanie macierzy pomyłek
    cms = [
        confusion_matrix(y_test, y_pred),
        confusion_matrix(y_test_sk, y_pred_sk),
        confusion_matrix(y_sms, y_pred_sms)
    ]

    # przygotowanie tytułów do macierzy pomyłek
    titles = [
        'Macierz pomyłek dla naiwnego klasyfikatora Bayesa "from scratch", na danych ze zbioru Telegram',
        'Macierz pomyłek dla naiwnego klasyfikatora Bayesa z biblioteki sklearn, na danych ze zbioru Telegram',
        'Macierz pomyłek dla naiwnego klasyfikatora Bayesa "from scratch", na danych testowych ze zbioru SMS'
    ]

    # pętla, która dla każdej z macierzy pomyłek rysuje tablicę
    for i, cm in enumerate(cms, start=1):
        plt.figure(i)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
        disp.plot(cmap='viridis', ax=plt.gca())
        plt.title(titles[i-1])

    plt.show()

    """
    Napisany przez nas algorytm dobrze wypada w porównaniu do tego zawartego w bibliotece sklrean; 88% dokładności algorytmu "from scratch" w porównaniu
    do 92% dokładności tego z biblioteki sklearn, podczas testowania na wiadomościach z bardzo obszernego zbioru Telegram. Podczas testów na innym, mniejszym
    zbiorze SMS, algorytm "from scratch", mimo treningu na zbiorze Telegram, osiąga dokładność nawet 96%.
    Początkowo trenowaliśmy oraz sprawdzaliśmy algorytm wyłącznie na danych ze zbioru SMS. Wtedy dokładność wynosiła 99% i była identyczna z tą osiąganą
    przez algorytm z biblioteki sklearn. Dopiero na większym zbiorze danych dostrzegalne są różnice w dokładności. Algorytm trenowany na dużym zbiorze osiąga
    również bardzo dobre wyniki, kiedy skorzysta się z niego do klasyfikacji wiadomości z mniejszego.
    """