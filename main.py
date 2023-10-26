#   IDEA    #
#   1.
#   zysk(czas) = liczba_uz * X + przychód * X + Zatrudnienie * X + B
#   X - czas (rok lub kwartały[1,2,3 ...])
#
#   2. SIMPLE
#   uzytkownicy (czas) = a1 * X + B

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lab1():
    exel_data = pd.read_excel("wykresik.xlsx", sheet_name="Arkusz1")

    kwartal_matrix = exel_data[["Kwartał"]].to_numpy()
    uz_matrix = exel_data["Liczba uzytkownikow"].to_numpy()

    kwartal_matrix_1 = np.insert(kwartal_matrix, 0, 1, axis=1)

    uz_matrix = np.array(uz_matrix).reshape(-1, 1)

    kwartal_matrix_t = np.transpose(kwartal_matrix_1)

    result_matrix = np.dot(np.dot(np.linalg.inv(np.dot(kwartal_matrix_t, kwartal_matrix_1)), kwartal_matrix_t), uz_matrix)

    print(result_matrix)

    draw(kwartal_matrix, uz_matrix, result_matrix[1][0], result_matrix[0][0])


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
    return a1*x + a0


if __name__ == "__main__":
    lab1()
