#   IDEA    #
#   1. PROPER
#   uzytkownicy (czas) = f1(czas)
#   przychod (uzytkownicy) = f2(uzytkownicy)
#   zatrudnienia (przychod) = f3(przychod)
#
#   2. SIMPLE
#   uzytkownicy (czas) = a1 * X + B

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#   zaraca a1 i a0 dla podanych danych
def create_model(X_table, Y_table):

    X_matrix = X_table.to_numpy()
    Y_matrix = Y_table.to_numpy()

    X_matrix_1 = np.insert(X_matrix, 0, 1, axis=1)

    Y_matrix = np.array(Y_matrix).reshape(-1, 1)

    X_matrix_t = np.transpose(X_matrix_1)

    result_matrix = np.dot(np.dot(np.linalg.inv(np.dot(X_matrix_t, X_matrix_1)), X_matrix_t), Y_matrix)

    print(result_matrix)

    return result_matrix[1][0], result_matrix[0][0]


def draw(x, y, a1, a0):
    plt.plot(x, y, label="data")

    y_func = func(a1, a0, x)
    plt.plot(x, y_func, label="func")

    plt.xlabel("Oś x")
    plt.ylabel("Oś y")
    plt.title("Wykres")
    plt.legend()

    plt.savefig("wykres.png")

    plt.show()


def func(a1, a0, x):
    return a1 * x + a0


if __name__ == "__main__":
    exel_data = pd.read_excel("dane.xlsx", sheet_name="Arkusz1")

    a1, a0 = create_model(exel_data[["Kwartał"]], exel_data["Liczba uzytkownikow"])

    draw(exel_data[["Kwartał"]].to_numpy(), exel_data[["Liczba uzytkownikow"]].to_numpy(), a1, a0)
