import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def load_data():
    data = pd.read_csv('../breast_cancer_wisconsin/data.csv', usecols=range(12))
    data = data.drop('id', axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    X = data.drop('diagnosis', axis=1)
    Y = data['diagnosis']

    # Usunięcie danych odstających (z wykorzystaniem metody IQR)
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    X = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
    Y = Y[X.index]  # Zaktualizowanie Y na podstawie indeksów z X

    # Normalizacja danych
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Uzupełnienie brakujących danych
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, Y_train, X_test, Y_test