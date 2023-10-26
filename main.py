#   IDEA    #
#   1. PROPER
#   zysk(czas) = liczba_uz * X + przychód * X + Zatrudnienie * X + B
#   X - czas (rok lub kwartały[1,2,3 ...])
#
#   2. SIMPLE
#   uzytkownicy (czas) = a1 * X + B

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Simple():
    exel_data = pd.read_excel("dane.xlsx", sheet_name="Arkusz1")

    kwartal_matrix = exel_data[["Kwartał"]].to_numpy()
    uz_matrix = exel_data["Liczba uzytkownikow"].to_numpy()

    kwartal_matrix_1 = np.insert(kwartal_matrix, 0, 1, axis=1)

    uz_matrix = np.array(uz_matrix).reshape(-1, 1)

    kwartal_matrix_t = np.transpose(kwartal_matrix_1)

    result_matrix = np.dot(np.dot(np.linalg.inv(np.dot(kwartal_matrix_t, kwartal_matrix_1)), kwartal_matrix_t), uz_matrix)

    print(result_matrix)

    draw(kwartal_matrix, uz_matrix, result_matrix[1][0], result_matrix[0][0])

def Proper():
    exel_data = pd.read_excel("dane.xlsx", sheet_name="Arkusz1")
    kwartal_matrix = exel_data[["Kwartał"]].to_numpy()
    kwartal_matrix_1 = np.insert(kwartal_matrix, 0, 1, axis=1)

    zat_matrix = exel_data[["Zatrudnienie"]].to_numpy()
    przy_matrix = exel_data[["Przychód"]].to_numpy()
    zysk_matrix = exel_data[["Zysk"]].to_numpy()
    uz_matrix = exel_data[["Liczba uzytkownikow"]].to_numpy()



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

#CHANGE HERE WHAT IDEA YOU WAHT TO COMPILE
if __name__ == "__main__":
    #Simple()
    Proper()
