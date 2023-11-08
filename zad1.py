#   IDEAS    #
#   uzytkownicy (czas) = f1(czas)               DONE
#   zatrudnienie (uzytkownicy) = f2(uzytkownicy)DONE
#   przychod (zatrudnienia) = f3(zatrudnienia)  DONE
#   prognoza zatrudnienia podając czas          DONE
#   sprawdzenie prognoz ze stroną internetową   TO DO
#   obliczenie parametrów                       DONE

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

podstawa_log = math.e


def create_model(X_table, Y_table, rodzaj):
    X_matrix = X_table.to_numpy()
    Y_matrix = Y_table.to_numpy()

    X_matrix = X_matrix[~np.isnan(X_matrix)]
    X_matrix = X_matrix.reshape(-1, 1)

    Y_matrix = Y_matrix[~np.isnan(Y_matrix)]
    Y_matrix = Y_matrix.reshape(-1, 1)

    X_matrix = X_matrix.astype(int)
    Y_matrix = Y_matrix.astype(int)

    X_matrix_1 = np.insert(X_matrix, 0, 1, axis=1)

    Y_matrix = np.array(Y_matrix).reshape(-1, 1)

    X_matrix_t = np.transpose(X_matrix_1)

    if rodzaj == "wykladniczy":

        Y_matrix_log = np.log(Y_matrix) / np.log(podstawa_log)

        result_matrix = np.dot(np.dot(np.linalg.inv(np.dot(X_matrix_t, X_matrix_1)), X_matrix_t), Y_matrix_log)

        a1, a0 = np.power(podstawa_log, result_matrix[1][0]), np.power(podstawa_log, result_matrix[0][0])

        predictions = func_liniowa(a1, a0, X_matrix[:, 0])

        print(f"Funkcja wykładnicza y(x) = {a0} + {a1}^x")

    elif rodzaj == "liniowy" or rodzaj == "liniowy2":

        result_matrix = np.dot(np.dot(np.linalg.inv(np.dot(X_matrix_t, X_matrix_1)), X_matrix_t), Y_matrix)

        a1, a0 = result_matrix[1][0], result_matrix[0][0]

        predictions = func_liniowa(a1, a0, X_matrix[:, 0])

        if rodzaj == "liniowy":
            print(f"Funkcja liniowa y(x) = {a1}*x + {a0}")
        else:
            print(f"Funkcja liniowa_2 y(x) = {a1}*x + {a0}")
    else:
        raise ValueError("Nieznany rodzaj modelu oczekiwano liniowego lub wykładniczego")

    if predictions is None:
        raise ValueError("Predictions nie zostało zdefiniowane")

    draw(X_matrix, Y_matrix, a1, a0, rodzaj)

    residuals = Y_matrix[:, 0] - predictions
    SSE = np.sum(residuals ** 2)
    MSE = SSE / (len(Y_matrix) - 2)
    stddev = np.sqrt(MSE)

    SST = np.sum((Y_matrix[:, 0] - np.mean(Y_matrix)) ** 2)
    R_squared = 1 - (SSE / SST)

    print(f"Odchylenie standardowe błędu: {stddev}")
    print(f"Współczynnik determinacji R^2: {R_squared}")

    return a1, a0, stddev, R_squared


def draw(x, y, a1, a0, rodzaj):
    plt.plot(x, y, label="data")

    if rodzaj == "liniowy":
        y_func = func_liniowa(a1, a0, x)
        plt.plot(x, y_func, label="func")
        plt.xlabel("Rok")
        plt.ylabel("Liczba użytkowników")

    elif rodzaj == "wykladniczy":
        y_func = func_wykl(a1, a0, x)
        plt.plot(x, y_func, label="func")
        plt.xlabel("Liczba użytkowników")
        plt.ylabel("Zatrudnienie")

    elif rodzaj == "liniowy2":
        y_func = func_liniowa(a1, a0, x)
        plt.plot(x, y_func, label="func")
        plt.xlabel("Zatrudnienie")
        plt.ylabel("Przychód")

    plt.title("Wykres")
    plt.legend()

    if rodzaj == "liniowy":
        plt.savefig("wykres_liniowy.png")
    elif rodzaj == "wykladniczy":
        plt.savefig("wykres_wykladniczy.png")
    elif rodzaj == "liniowy2":
        plt.savefig("wykres_linowy_2.png")

    plt.show()


def func_liniowa(a1, a0, x):
    return a1 * x + a0


def func_wykl(a1, a0, x):
    return a0 * np.power(a1, x)


if __name__ == "__main__":
    #   ZAŁADOWANIE EXEL
    exel_data = pd.read_excel("dane.xlsx", sheet_name="Arkusz1")

    #   TWORZENIE MODELI
    #   LICZBA_UZ (CZAS) - FUNKCJA LINIOWA
    a1_lin, a0_lin, stddev_lin, r2_lin = create_model(exel_data[["Rok"]],
                                                      exel_data["Uzytkownicy-Rok"], "liniowy")

    #   PRZYCHOD (LICZBA_UZ) - FUNKCJA WYKLADNICZA
    a1_wyk, a0_wyk, stdedev_wyk, r2_wyk = create_model(exel_data[["Uzytkownicy-Rok"]],
                                                       exel_data["Zatrudnienie-Rok"], "wykladniczy")

    #   ZATRUDNIENIE (PRZYCHOD) - FUNKCJA LINIOWA
    a1_lin_2, a0_lin_2, stdedev_lin_2, r2_lin_2 = create_model(exel_data[["Zatrudnienie-Rok"]],
                                                               exel_data["Przychod-Rok"], "liniowy2")

    #   PRZYCHOD POJEDYNCZY
    print("PRZEWIDYWANIE POJEDYNCZE")
    #   LICZBA UZYTKOWNIKOW W PRZYSZLOSCI DLA ROKU 2019
    print("UZYTKOWNICY W ROKU 2019: ", func_liniowa(a1_lin, a0_lin, 2019))

    #   ZATRUDNIENIE NA PODSTAWIE UZYTKOWNIKÓW DLA LICZBY_UZYTKOWNIKÓW = 9000
    print("ZATRUDNIENIE DLA LICZ. UZYTKOWNIKOW 9000: ", func_wykl(a1_wyk, a0_wyk, 9000))

    #   PRZYCHÓD NA PODSTAWIE ZATRUDNIENIA DLA ZATRUDNIENIA = 30000
    print("PRZYCHÓD NA PODSTAWIE ZATRUDNIENIA 30 000: ", func_liniowa(a1_lin_2, a0_lin_2, 30000))

    #   ZATRUDNIENIE ŁĄCZONE
    print("PRZEWIDYWANIE PODWÓJNE")
    #   PRZEWIDYWANE ZATRUDNIENIE NA PODSTAWIE ROKU 2020
    print("ZATRUDNIENIE NA PODSTAWIE ROKU 2020: ", func_wykl(a1_wyk, a0_wyk,
                                                             func_liniowa(a1_lin, a0_lin, 2020)))

    #   PRZYCHÓD ŁĄCZONY
    #   PRZEWIDYWANY PRZYCHÓD NA PODSTAWIE LICZBY UZYTKOWNIKOW
    print("PRZYCHOD NA PODSTAWIE LICZ. UŻYTKOWNIKÓW: ", func_liniowa(a1_lin_2, a0_lin_2,
                                                                     func_wykl(a1_wyk, a0_wyk, 9200)))

    #   PRZEWIYWANIE POTRÓJNE
    print("!!!PRZYCHÓD DLA ROKU 2019!!!", func_liniowa(a1_lin_2, a0_lin_2,
                                                       func_wykl(a1_wyk, a0_wyk,
                                                        func_liniowa(a1_lin, a0_lin, 2019))))