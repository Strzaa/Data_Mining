#   IDEA    #
#   1. PROPER
#   uzytkownicy (czas) = f1(czas)               DONE
#   przychod (uzytkownicy) = f2(uzytkownicy)    DONE
#   zatrudnienia (przychod) = f3(przychod)      TO DO
#   prognoza zatrudnienia podając czas          TO DO
#   sprawdzenie prognoz ze stroną internetową   TO DO
#   obliczenie parametrów                       TO DO
#
#   2. SIMPLE
#   uzytkownicy (czas) = a1 * X + B             DONE

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_model(X_table, Y_table, rodzaj):
    X_matrix = X_table.to_numpy()
    Y_matrix = Y_table.to_numpy()

    X_matrix_1 = np.insert(X_matrix, 0, 1, axis=1)

    Y_matrix = np.array(Y_matrix).reshape(-1, 1)

    X_matrix_t = np.transpose(X_matrix_1)

    if rodzaj == "wykladniczy":
        Y_matrix_log = np.log(Y_matrix) / np.log(math.e)

        result_matrix = np.dot(np.dot(np.linalg.inv(np.dot(X_matrix_t, X_matrix_1)), X_matrix_t), Y_matrix_log)

        a1, a0 = np.power(math.e, result_matrix[1][0]), np.power(math.e, result_matrix[0][0])

        print(f"Funkcja wykładnicza y(x) = {a0} + {a1}^x")

        draw(X_matrix, Y_matrix, a1, a0, rodzaj)

        return

    elif rodzaj == "liniowy":
        result_matrix = np.dot(np.dot(np.linalg.inv(np.dot(X_matrix_t, X_matrix_1)), X_matrix_t), Y_matrix)

        a1, a0 = result_matrix[1][0], result_matrix[0][0]

        print(f"Funkcja liniowa y(x) = {a1}*x + {a0}")

        draw(X_matrix, Y_matrix, a1, a0, rodzaj)

        return


def draw(x, y, a1, a0, rodzaj):
    plt.plot(x, y, label="data")

    if rodzaj == "liniowy":
        y_func = func_liniowa(a1, a0, x)
        plt.plot(x, y_func, label="func")

    elif rodzaj == "wykladniczy":
        y_func = func_wykl(a1, a0, x)
        plt.plot(x, y_func, label="func")

    plt.xlabel("Oś x")
    plt.ylabel("Oś y")
    plt.title("Wykres")
    plt.legend()

    if rodzaj == "liniowy":
        plt.savefig("wykres_liniowy.png")
    elif rodzaj == "wykladniczy":
        plt.savefig("wykres_wykladniczy.png")

    plt.show()


def func_liniowa(a1, a0, x):
    return a1 * x + a0


def func_wykl(a1, a0, x):
    return a0 * np.power(a1, x)


if __name__ == "__main__":
    exel_data = pd.read_excel("dane.xlsx", sheet_name="Arkusz1")

    create_model(exel_data[["Kwartał"]], exel_data["Liczba uzytkownikow"], "liniowy")

    create_model(exel_data[["Liczba uzytkownikow"]], exel_data["Przychod"], "wykladniczy")
