import pandas as pd
import string
from sklearn.model_selection import train_test_split

def load_dataset(datapath, sep, names):
    
    # wczytanie pliku z danymi według wskazanych separatorów i docelowych nazw kolumn
    data = pd.read_csv(datapath, sep=sep, header=None, names=names)
    return data

# Do użycia w funkcji przyjmującej tekst jako listę słów
def text_into_list(text):
    
    # usunięcie interpunkcji
    text = text.translate(str.maketrans('', '',string.punctuation))
    # konwersja na małe litery
    text = text.lower()

    # podział łańcucha znaków na listę słów
    return list(text.split())

# Do użycia funkcji przyjmującej tekst jako string
def text_into_string(text):

    # usunięcie interpunkcji
    text = text.translate(str.maketrans('', '', string.punctuation))
    # konwersja na małe litery
    text = text.lower()

    # tworzenie nowego ciągu zawierającego słowa oddzielone spacją
    # zwracamy string zamiast listy dla późniejszej wektoryzacji
    return ' '.join(text.split())


def prepare_data(data, cleaning_function, names):
    
    # stworzenie kolumny z tekstem wyczyszczonym przy pomocy przekazanej funkcji
    data['words'] = data[names[1]].apply(cleaning_function)

    # podział na kolumnę etykiet oraz dane tekstowe
    X, y = data.iloc[:,1:], data.iloc[:,0]
    # podział na dane testowe i treningowe
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)

    # reset indeksów w nowych dataframe'ach
    for df in [X_train, X_test, y_train, y_test]:
        df.reset_index(drop=True, inplace=True)

    return (X_train, X_test, y_train, y_test)