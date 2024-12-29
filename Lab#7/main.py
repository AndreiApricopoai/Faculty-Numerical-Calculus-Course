import numpy as np
import cmath
import random


def calculate_R(coeffs):
    a0 = coeffs[0]
    max_coeff = max(abs(coeff) for coeff in coeffs[1:])
    return (abs(a0) + max_coeff) / abs(a0)


def horner(coeffs, x):
    result = coeffs[0]
    for coeff in coeffs[1:]:
        result = result * x + coeff
    return result


def muller_method(coeffs, x0, x1, x2, tol, max_iter):
    h0 = x1 - x0
    h1 = x2 - x1
    delta0 = (horner(coeffs, x1) - horner(coeffs, x0)) / h0
    delta1 = (horner(coeffs, x2) - horner(coeffs, x1)) / h1
    d = (delta1 - delta0) / (h1 + h0)
    k = 3
    x = None

    while k <= max_iter:
        b = delta1 + h1 * d
        c = horner(coeffs, x2)
        discriminant = b**2 - 4 * c * d

        if discriminant < 0:
            print("Encountered complex root.")
            return None  # Return None if a complex root is encountered

        b_sign = np.sign(b) if b != 0 else 1
        sqrt_discriminant = cmath.sqrt(discriminant)
        denom = b + b_sign * sqrt_discriminant

        if abs(denom) < tol:
            print("Denominator too small, stopping iteration.")
            return x

        Delta_x = 2 * c / denom
        x = x2 - Delta_x

        if abs(Delta_x) < tol:
            return x

        # Prepare for the next iteration
        x0, x1, x2 = x1, x2, x
        h0 = x1 - x0
        h1 = x2 - x1
        delta0 = (horner(coeffs, x1) - horner(coeffs, x0)) / h0
        delta1 = (horner(coeffs, x2) - horner(coeffs, x1)) / h1
        d = (delta1 - delta0) / (h1 + h0)
        k += 1

    return x


def main():
    coeffs_list = [
        [1, -6, 11, -6],
        [1.0 / 42, -55.0 / 42, -42.0 / 42, 49.0 / 42, -6.0 / 42],
        [1.0 / 8, -38.0 / 8, 49.0 / 8, -22.0 / 8, 3.0 / 8],
        [1, -6, 13, -12, 4]
    ]

    for coeffs in coeffs_list:
        R = calculate_R(coeffs)
        print(f"\nInterval for polynomial with coefficients {coeffs} is [-{R}, {R}]")
        for i in range(5):
            x0, x1, x2 = [random.uniform(-R, R) for _ in range(3)]
            root = muller_method(coeffs, x0, x1, x2, 1e-6, 100)
            print(
                f"{i + 1} Root for polynomial with coefficients {coeffs} is approximately {root}, started from x0={x0}, x1={x1}, x2={x2}")


if __name__ == "__main__":
    main()
