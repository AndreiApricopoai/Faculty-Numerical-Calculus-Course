import numpy as np

precision = 16
eps = 10 ** (-precision)

def read_matrix(path):
    with open(path) as f:
        lines = f.readlines()

    # The first line contains the size of the matrix.
    n = int(lines[0].strip())

    # Initialize an empty dictionary for each row in the matrix.
    mp = [{} for _ in range(n)]

    # Process each line after the first one.
    for line in lines[1:]:
        try:
            # Attempt to unpack the three expected values.
            val, i, j = line.split(',')
        except ValueError:
            # Skip lines that do not have three values.
            continue

        # Convert the string values to appropriate types.
        val = float(val.strip())
        i = int(i.strip())
        j = int(j.strip())

        # Check for and handle duplicate entries by summing the values.
        if mp[i].get(j) is not None:
            mp[i][j] += val
        else:
            mp[i][j] = val

        # Check for non-zero diagonal elements.
        if i == j and abs(val) < eps:
            print(f'Null element on main diagonal at position {i}')
            break

    return n, mp


def read_line_matrix(path, n):
    matrix = np.zeros((n, 1), dtype='float32')

    with open(path) as f:
        lines = f.readlines()

    line_index = 0  # Index for the lines being processed.
    for i in range(1, len(lines)):  # Skip the first line, which contains the dimension.
        line = lines[i].strip()  # Remove any leading/trailing whitespace.
        if line:  # Check if the line is not empty.
            try:
                matrix[line_index, 0] = float(line)
                line_index += 1
            except ValueError:
                print(f"Unable to convert line {i} to float: {line}")
                continue

    return matrix


def gauss_seidel_method(a, b, eps):
    n = len(a)
    x = np.zeros((n, 1), dtype='float32')

    for iteration in range(10000):
        x_new = x.copy()
        for i in range(n):
            if i not in a[i]:  # Ensure the diagonal element exists
                print(f"No diagonal element at row {i}")
                return x
            sum1 = sum(a[i][j] * x_new[j, 0] for j in range(i) if j in a[i])
            sum2 = sum(a[i][j] * x[j, 0] for j in range(i + 1, n) if j in a[i])
            x_new[i, 0] = (b[i, 0] - sum1 - sum2) / a[i][i]  # Access b[i, 0] instead of b[i]

        diff = np.linalg.norm(x_new - x)
        if diff < eps:
            print(f'Solution found in {iteration} iterations')
            break

        x = x_new

    return x

# Main function adapted for testing with one matrix and vector
def main():
    n_a1, a1 = read_matrix('a_1.txt')
    b1 = read_line_matrix('b_1.txt', n_a1)
    x = gauss_seidel_method(a1, b1, eps)
    print(x)

if __name__ == '__main__':
    main()
