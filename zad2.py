#   IDEAS   #
#   MODEL WIELOKRYTERIALNY
#   Y(X1, ..., X13) = A0 + A1X1 + ... + A13X13
#   Y(X1, ..., X13) - Pct.BF

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

liczba_obs = 250


def create_model(X_matrix, Y_table) -> np.array:
    Y_matrix = Y_table.to_numpy()

    X_matrix_1 = np.insert(X_matrix, 0, 1, axis=1)

    Y_matrix = np.array(Y_matrix).reshape(-1, 1)

    X_matrix_t = np.transpose(X_matrix_1)

    result_matrix = np.dot(np.dot(np.linalg.inv(np.dot(X_matrix_t, X_matrix_1)), X_matrix_t), Y_matrix)

    return result_matrix


def obliczenie_wynikow(list_a):
    wyniki = []
    wynik = 0
    for iter in range(0, liczba_obs):
        for element in range(len(list_a)-1):
            wynik += list_a[element+1][0] * X_matrix[iter][element]
        wynik += list_a[0][0]
        wyniki.append(wynik)
        wynik = 0
    return wyniki


def draw(x, y):
    plt.plot(x, y, label="data")

    plt.xlabel("Oś X")
    plt.ylabel("Oś Y")
    plt.title("Wykres")
    plt.legend()

    plt.savefig("wykres_zad2.png")

    plt.show()


def r_2(wyniki, y_data):
    y_sr = y_data.mean()

    licznik = 0
    mianownik = 0
    for iter in range(0, liczba_obs):
        licznik += ((y_data[iter] - wyniki[iter]) ** 2)
        mianownik += ((y_data[iter] - y_sr) ** 2)
    return 1 - (licznik / mianownik)


if __name__ == "__main__":
    #   ZAŁADOWANIE EXEL
    exel_data = pd.read_excel("dane.xlsx", sheet_name="Arkusz2")

    X_matrix = exel_data[["Density", "Age", "Weight", "Height", "Neck", "Chest", "Abdomen", "Waist", "Hip",
                                     "Thigh", "Knee", "Ankle", "Bicep", "Forearm", "Wrist"]].to_numpy()

    list_a = create_model(X_matrix, exel_data[["Pct.BF"]])

    # print("Wartości wspolczynnikow a: ", list_a)

    wyniki=obliczenie_wynikow(list_a)

    R_2 = r_2(wyniki, exel_data[["Pct.BF"]].to_numpy())

    print("R^2 obliczone recznie: ", R_2, " R^2 przy uzyciu biblioteki: ", r2_score(exel_data[["Pct.BF"]].to_numpy(),
                                                                                    wyniki))

    print("R^2 skorygowane: ", 1 - (1 - R_2) * (liczba_obs - 1)/(liczba_obs - len(list_a) - 1 - 1))


