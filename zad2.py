#   IDEAS   #
#   MODEL WIELOKRYTERIALNY
#   Y(X1, ..., X13) = A0 + A1X1 + ... + A13X13
#   Y(X1, ..., X13) - Pct.BF

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_model(X_matrix, Y_table) -> np.array:
    Y_matrix = Y_table.to_numpy()

    X_matrix_1 = np.insert(X_matrix, 0, 1, axis=1)

    Y_matrix = np.array(Y_matrix).reshape(-1, 1)

    X_matrix_t = np.transpose(X_matrix_1)

    result_matrix = np.dot(np.dot(np.linalg.inv(np.dot(X_matrix_t, X_matrix_1)), X_matrix_t), Y_matrix)

    return result_matrix


def draw(x, y):
    plt.plot(x, y, label="data")

    plt.xlabel("Oś X")
    plt.ylabel("Oś Y")
    plt.title("Wykres")
    plt.legend()

    plt.savefig("wykres_zad2.png")

    plt.show()


if __name__ == "__main__":
    #   ZAŁADOWANIE EXEL
    exel_data = pd.read_excel("dane.xlsx", sheet_name="Arkusz2")

    X_matrix = exel_data[["Density", "Age", "Weight", "Height", "Neck", "Chest", "Abdomen", "Waist", "Hip",
                                     "Thigh", "Knee", "Ankle", "Bicep", "Forearm", "Wrist"]].to_numpy()

    list_a = create_model(X_matrix, exel_data[["Pct.BF"]])

    wyniki = []
    wynik = 0
    for iter in range(0, 10):
        for element in range(len(list_a)-1):
            wynik += list_a[element+1][0] * X_matrix[iter][element]
        wynik += list_a[0][0]
        wyniki.append(wynik)
        wynik = 0

    print(wyniki)