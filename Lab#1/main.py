import random
import math
import matplotlib.pyplot as plt
import numpy as np

'''
Să se găsească cel mai mic număr pozitiv u > 0, de forma u = 10-m care satisface
proprietatea:
1 1 + ≠ c u
unde prin +c am notat operația de adunare efectuată de calculator. Numărul u se
numește precizia mașină
'''


def ex_1():
    m = 1
    while 1 + 10 ** (-m) != 1:
        m += 1
    return m - 1


'''
Operația +c este neasociativă: fie numerele x=1.0 , y = u , z = u , unde u este precizia
mașină calculată anterior. Să se verifice că operația de adunare efectuată de calculator
este neasociativă:
(x +c y) +c z ≠ x +c (y +c z
'''


def ex_2():
    u = 10 ** (-ex_1())
    x = 1.0
    y = u
    z = u
    if (x + y) + z != x + (y + z):
        print("Adunarea este neasociativa")
    else:
        print("Adunarea este asociativa")

    y = 10 ** (-ex_1())
    z = 10 ** (-ex_1())
    x = random.uniform(y, z)
    while (x * y) * z == x * (y * z):
        x = x * 10

    print("x = ", x)
    print("y = ", y)
    print("z = ", z)
    print((x * y) * z)
    print(x * (y * z))


def T_1(a):
    return a


def T_2(a):
    return (3 * a) / (3 - a ** 2)


def T_3(a):
    return (15 * a - a ** 3) / (15 - 6 * a ** 2)


def T_4(a):
    return (105 * a - 10 * a ** 3) / (105 - 45 * a ** 2 + a ** 4)


def T_5(a):
    return (945 * a - 105 * a ** 3 + a ** 5) / (945 - 420 * a ** 2 + 15 * a ** 4)


def T_6(a):
    return (10395 * a - 1620 * a ** 3 + 21 * a ** 5) / (10395 - 4725 * a ** 2 + 210 * a ** 4 - a ** 6)


def T_7(a):
    return (135135 * a - 17325 * a ** 3 + 378 * a ** 5 - a ** 7) / (
                135135 - 62370 * a ** 2 + 3150 * a ** 4 - 28 * a ** 6)


def T_8(a):
    return (2027025 * a - 270270 * a ** 3 + 6930 * a ** 5 - 36 * a ** 7) / (
                2027025 - 945945 * a ** 2 + 51975 * a ** 4 - 630 * a ** 6 + a ** 8)


def T_9(a):
    return (34459425 * a - 4729725 * a ** 3 + 135135 * a ** 5 - 990 * a ** 7 + a ** 9) / (
                34459425 - 16216200 * a ** 2 + 945945 * a ** 4 - 13860 * a ** 6 + 45 * a ** 8)


def tg(x):
    return math.tan(x)


def ex_3():
    random_numbers = []
    for i in range(0, 10000):
        random_numbers.append(random.uniform(-(math.pi / 2), math.pi / 2))

    differences_tan = {i: [] for i in range(9)}
    differences_sin = {i: [] for i in range(9)}
    differences_cos = {i: [] for i in range(9)}

    for j in range(0, 10000):
        differences_tan[0].append(abs(T_1(random_numbers[j]) - tg(random_numbers[j])))
        differences_tan[1].append(abs(T_2(random_numbers[j]) - tg(random_numbers[j])))
        differences_tan[2].append(abs(T_3(random_numbers[j]) - tg(random_numbers[j])))
        differences_tan[3].append(abs(T_4(random_numbers[j]) - tg(random_numbers[j])))
        differences_tan[4].append(abs(T_5(random_numbers[j]) - tg(random_numbers[j])))
        differences_tan[5].append(abs(T_6(random_numbers[j]) - tg(random_numbers[j])))
        differences_tan[6].append(abs(T_7(random_numbers[j]) - tg(random_numbers[j])))
        differences_tan[7].append(abs(T_8(random_numbers[j]) - tg(random_numbers[j])))
        differences_tan[8].append(abs(T_9(random_numbers[j]) - tg(random_numbers[j])))

        differences_sin[0].append(abs(T_1_sin(random_numbers[j]) - sin(random_numbers[j])))
        differences_sin[1].append(abs(T_2_sin(random_numbers[j]) - sin(random_numbers[j])))
        differences_sin[2].append(abs(T_3_sin(random_numbers[j]) - sin(random_numbers[j])))
        differences_sin[3].append(abs(T_4_sin(random_numbers[j]) - sin(random_numbers[j])))
        differences_sin[4].append(abs(T_5_sin(random_numbers[j]) - sin(random_numbers[j])))
        differences_sin[5].append(abs(T_6_sin(random_numbers[j]) - sin(random_numbers[j])))
        differences_sin[6].append(abs(T_7_sin(random_numbers[j]) - sin(random_numbers[j])))
        differences_sin[7].append(abs(T_8_sin(random_numbers[j]) - sin(random_numbers[j])))
        differences_sin[8].append(abs(T_9_sin(random_numbers[j]) - sin(random_numbers[j])))

        differences_cos[0].append(abs(T_1_cos(random_numbers[j]) - cos(random_numbers[j])))
        differences_cos[1].append(abs(T_2_cos(random_numbers[j]) - cos(random_numbers[j])))
        differences_cos[2].append(abs(T_3_cos(random_numbers[j]) - cos(random_numbers[j])))
        differences_cos[3].append(abs(T_4_cos(random_numbers[j]) - cos(random_numbers[j])))
        differences_cos[4].append(abs(T_5_cos(random_numbers[j]) - cos(random_numbers[j])))
        differences_cos[5].append(abs(T_6_cos(random_numbers[j]) - cos(random_numbers[j])))
        differences_cos[6].append(abs(T_7_cos(random_numbers[j]) - cos(random_numbers[j])))
        differences_cos[7].append(abs(T_8_cos(random_numbers[j]) - cos(random_numbers[j])))
        differences_cos[8].append(abs(T_9_cos(random_numbers[j]) - cos(random_numbers[j])))

    means_tan = []
    means_sin = []
    means_cos = []

    for i in range(9):
        means_tan.append(np.mean(differences_tan[i]))
        means_sin.append(np.mean(differences_sin[i]))
        means_cos.append(np.mean(differences_cos[i]))

    min_mean_tan = min(means_tan)
    min_mean_sin = min(means_sin)
    min_mean_cos = min(means_cos)

    print("Best approximation  for tan: T_", means_tan.index(min_mean_tan))
    mean_dictionary_tan = {i: means_tan[i] for i in range(9)}
    sorted_means_tan = sorted(mean_dictionary_tan.items(), key=lambda x: x[1])
    print("Ordered approximation functions for tan: ")
    for i in range(9):
        print("T_", sorted_means_tan[i][0], ":", sorted_means_tan[i][1])

    print("Best approximation  for sin: T_", means_sin.index(min_mean_sin))
    mean_dictionary_sin = {i: means_sin[i] for i in range(9)}
    sorted_means_sin = sorted(mean_dictionary_sin.items(), key=lambda x: x[1])
    print("Ordered approximation functions for sin: ")
    for i in range(9):
        print("T_", sorted_means_sin[i][0], ":", sorted_means_sin[i][1])

    print("Best approximation  for cos: T_", means_cos.index(min_mean_cos))
    mean_dictionary_cos = {i: means_cos[i] for i in range(9)}
    sorted_means_cos = sorted(mean_dictionary_cos.items(), key=lambda x: x[1])
    print("Ordered approximation functions for cos: ")
    for i in range(9):
        print("T_", sorted_means_cos[i][0], ":", sorted_means_cos[i][1])

    plt.plot(range(1, 10), means_tan, label="tan")
    plt.plot(range(1, 10), means_sin, label="sin")
    plt.plot(range(1, 10), means_cos, label="cos")
    plt.xlabel("Approximation function")
    plt.ylabel("Mean")
    plt.title("Mean of the differences between the approximation functions and the math functions")
    plt.legend()
    plt.scatter(means_tan.index(min_mean_tan) + 1, min_mean_tan, color="red")
    plt.scatter(means_sin.index(min_mean_sin) + 1, min_mean_sin, color="red")
    plt.scatter(means_cos.index(min_mean_cos) + 1, min_mean_cos, color="red")
    plt.show()


# BONUS
def sin(x):
    return math.sin(x)


def cos(x):
    return math.cos(x)


# Transofrm the T_x for tg to T_x_sin for sin and T_x_cos for cos
def T_1_sin(a):
    return (1 - T_1((2 * a - math.pi) / 4) ** 2) / (1 + T_1((2 * a - math.pi) / 4) ** 2)


def T_2_sin(a):
    return (1 - T_2((2 * a - math.pi) / 4) ** 2) / (1 + T_2((2 * a - math.pi) / 4) ** 2)


def T_3_sin(a):
    return (1 - T_3((2 * a - math.pi) / 4) ** 2) / (1 + T_3((2 * a - math.pi) / 4) ** 2)


def T_4_sin(a):
    return (1 - T_4((2 * a - math.pi) / 4) ** 2) / (1 + T_4((2 * a - math.pi) / 4) ** 2)


def T_5_sin(a):
    return (1 - T_5((2 * a - math.pi) / 4) ** 2) / (1 + T_5((2 * a - math.pi) / 4) ** 2)


def T_6_sin(a):
    return (1 - T_6((2 * a - math.pi) / 4) ** 2) / (1 + T_6((2 * a - math.pi) / 4) ** 2)


def T_7_sin(a):
    return (1 - T_7((2 * a - math.pi) / 4) ** 2) / (1 + T_7((2 * a - math.pi) / 4) ** 2)


def T_8_sin(a):
    return (1 - T_8((2 * a - math.pi) / 4) ** 2) / (1 + T_8((2 * a - math.pi) / 4) ** 2)


def T_9_sin(a):
    return (1 - T_9((2 * a - math.pi) / 4) ** 2) / (1 + T_9((2 * a - math.pi) / 4) ** 2)


def T_1_cos(a):
    return (1 - T_1(a / 2) ** 2) / (1 + T_1(a / 2) ** 2)


def T_2_cos(a):
    return (1 - T_2(a / 2) ** 2) / (1 + T_2(a / 2) ** 2)


def T_3_cos(a):
    return (1 - T_3(a / 2) ** 2) / (1 + T_3(a / 2) ** 2)


def T_4_cos(a):
    return (1 - T_4(a / 2) ** 2) / (1 + T_4(a / 2) ** 2)


def T_5_cos(a):
    return (1 - T_5(a / 2) ** 2) / (1 + T_5(a / 2) ** 2)


def T_6_cos(a):
    return (1 - T_6(a / 2) ** 2) / (1 + T_6(a / 2) ** 2)


def T_7_cos(a):
    return (1 - T_7(a / 2) ** 2) / (1 + T_7(a / 2) ** 2)


def T_8_cos(a):
    return (1 - T_8(a / 2) ** 2) / (1 + T_8(a / 2) ** 2)


def T_9_cos(a):
    return (1 - T_9(a / 2) ** 2) / (1 + T_9(a / 2) ** 2)


def main():
    print("Exercitiul 1: ", ex_1())
    print("Exercitiul 2: ")
    ex_2()
    print("Exercitiul 3: ")
    ex_3()


if __name__ == "__main__":
    main()
