#   IDEAS   #
#   MODEL WIELOKRYTERIALNY
#   Y(X1, ..., X13) = A0 + A1X1 + ... + A13X13
#   Y(X1, ..., X13) - Pct.BF
#   Zrobi najpierw X z Y a pozniej X z X. Bierzemy pod uwage r graniczne

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import t


def create_model(X_matrix_df, Y_table, licznik):
    ###
    X_matrix = X_matrix_df.to_numpy()
    Y_matrix = Y_table.to_numpy()

    X_matrix = X_matrix.astype(float)
    Y_matrix = Y_matrix.astype(float)

    X_matrix_1 = np.insert(X_matrix, 0, 1, axis=1)

    Y_matrix = np.array(Y_matrix).reshape(-1, 1)

    X_matrix_t = np.transpose(X_matrix_1)

    result_matrix = np.dot(np.dot(np.linalg.inv(np.dot(X_matrix_t, X_matrix_1)), X_matrix_t), Y_matrix)

    wyniki = obliczenie_wynikow(result_matrix, X_matrix)

    R_2 = r_2(wyniki, Y_matrix)

    print(f"MODEL_{licznik} R^2 obliczone: {np.round(R_2, 10)}")

    R_2_better = 1 - (1 - R_2) * (len(X_matrix) - 1) / (len(X_matrix) - len(result_matrix) - 1 - 1)
    print(f"MODEL_{licznik} R^2 skorygowane: {np.round(R_2_better, 10)}")

    # R_2_bib = r2_score(Y_matrix, np.dot(X_matrix_1, result_matrix))
    # print("Biblioteka: ", R_2_bib)

    return result_matrix, licznik + 1


def obliczenie_wynikow(list_a, X_matrix):
    wyniki = []
    wynik = 0
    for iter in range(0, len(X_matrix)):
        for element in range(0, len(list_a) - 1):
            wynik += list_a[element + 1][0] * X_matrix[iter][element]
        wynik += list_a[0][0]
        wyniki.append(wynik)
        wynik = 0
    return wyniki


def r_2(wyniki, y_data):
    y_sr = y_data.mean()

    licznik = 0
    mianownik = 0
    for iter in range(0, len(y_data)):
        licznik += ((y_data[iter] - wyniki[iter]) ** 2)
        mianownik += ((y_data[iter] - y_sr) ** 2)
    return 1 - (licznik / mianownik)


def korelacja(data, licznik):
    X_df = pd.DataFrame(data)

    nazwy_kolumn = X_df.columns.tolist()

    liczba_kolumn = X_df.shape[1]

    corr_m = np.zeros((liczba_kolumn, liczba_kolumn))

    for i in range(liczba_kolumn):
        for j in range(liczba_kolumn):
            korelacja_ij = X_df.iloc[:, i].corr(X_df.iloc[:, j])
            corr_m[i][j] = korelacja_ij

    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(corr_m, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=nazwy_kolumn,
                          yticklabels=nazwy_kolumn)
    plt.title(f'Korelacja między zmiennymi_{licznik}')
    plt.show()
    return licznik + 1


def podzial(data):
    liczba = len(data)

    ostatnie_20 = data.iloc[liczba - 20 :]
    pozostala_czesc = data.iloc[: liczba - 20]

    return pozostala_czesc, ostatnie_20


def sprawdzenie(exel_data_test, list_a_4, name):
    data_no_y = np.array(exel_data_test.drop([name], axis=1))
    data = pd.DataFrame(exel_data_test)
    wyniki = obliczenie_wynikow(list_a_4, data_no_y)
    sr_blad = 0
    for y in range(0, len(wyniki)):
        sr_blad += (data.loc[y + 230, name] - wyniki[y]) ** 2

    return math.sqrt(sr_blad)


def oblicz_r(data):
    t_value = t.ppf(0.05, len(data))
    return math.sqrt((t_value ** 2)/(t_value ** 2 + len(data) - 2))


if __name__ == "__main__":
    licznik_model = 0
    licznik_corr = 0

    #   ZAŁADOWANIE EXEL
    exel_data_raw = pd.read_excel("dane.xlsx", sheet_name="Arkusz2")

    exel_data, exel_data_test = podzial(exel_data_raw)

    # R_graniczne dla a = 0.05
    r_graniczne = oblicz_r(exel_data_raw)
    print("R_graniczne: ", r_graniczne)

    list_a_1, licznik_model = create_model(exel_data.drop(["Pct.BF"], axis=1), exel_data[["Pct.BF"]], licznik_model)

    licznik_corr = korelacja(exel_data, licznik_corr)
    # XY
    # usuwam Height
    exel_data_2 = exel_data.drop(["Height"], axis=1)
    licznik_corr = korelacja(exel_data_2, licznik_corr)

    # XX
    # usuwam Age, Ankle, Wrist, Forearm
    exel_data_3 = exel_data_2.drop(["Age", "Ankle", "Wrist", "Forearm"], axis=1)
    licznik_corr = korelacja(exel_data_3, licznik_corr)

    # usuwam Bicep, Knee, Neck, Thigh
    exel_data_4 = exel_data_3.drop(["Bicep", "Knee", "Neck", "Thigh"], axis=1)
    licznik_corr = korelacja(exel_data_4, licznik_corr)

    # usuwam Hip, Waist, Adbomen, Chest, Weight
    exel_data_5 = exel_data_4.drop(["Hip", "Waist", "Abdomen", "Chest", "Weight"], axis=1)

    list_a_2, licznik_model = create_model(exel_data_5.drop(["Pct.BF"], axis=1), exel_data[["Pct.BF"]], licznik_model)

    # sprawdzam model
    print(list_a_2)

    # obliczenie blędów
    sredni_blad = sprawdzenie(exel_data_test, list_a_1, "Pct.BF")
    print("Średni bląd modelu surowego: ", sredni_blad)

    sredni_blad = sprawdzenie(exel_data_test[["Density", "Pct.BF"]], list_a_2, "Pct.BF")
    print("Średni bląd modelu przygotowanego: ", sredni_blad)

    ################# BICEP #################################

    exel_data, exel_data_test = podzial(exel_data_raw)

    licznik_model = 10
    licznik_corr = 10

    licznik_corr = korelacja(exel_data, licznik_corr)

    list_a_1, licznik_model = create_model(exel_data.drop(["Bicep"], axis=1), exel_data[["Bicep"]], licznik_model)

    # XY
    # usuwam Age
    exel_data_2 = exel_data.drop(["Age"], axis=1)
    licznik_corr = korelacja(exel_data_2, licznik_corr)

    # XX
    # usuwam Height, Ankle, Pct.BF, Density
    exel_data_3 = exel_data_2.drop(["Height","Ankle","Pct.BF","Density"], axis=1)
    licznik_corr = korelacja(exel_data_3, licznik_corr)

    # usuwam Wrist, Abdomen, Waist, Neck
    exel_data_4 = exel_data_3.drop(["Wrist","Abdomen","Waist","Neck"], axis=1)
    licznik_corr = korelacja(exel_data_4, licznik_corr)

    list_a_2, licznik_model = create_model(exel_data_4.drop(["Bicep"], axis=1), exel_data[["Bicep"]], licznik_model)

    # usuwam Forearm, Knee
    exel_data_5 = exel_data_4.drop(["Forearm", "Knee"], axis=1)
    licznik_corr = korelacja(exel_data_5, licznik_corr)

    list_a_3, licznik_model = create_model(exel_data_5.drop(["Bicep"], axis=1), exel_data[["Bicep"]], licznik_model)

    # zostawiam MODEL_11
    # sprawdam model
    print(list_a_2)

    # obliczenie blędów
    sredni_blad = sprawdzenie(exel_data_test, list_a_1, "Bicep")
    print("Średni bląd modelu surowego: ", sredni_blad)

    sredni_blad = sprawdzenie(exel_data_test[["Weight","Chest","Hip","Thigh","Knee","Forearm","Bicep"]], list_a_2, "Bicep")
    print("Średni bląd modelu przygotowanego: ", sredni_blad)