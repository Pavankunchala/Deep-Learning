v = [5, 10, 2, 6, 1]

mv = [
    [5, 10, 2, 6, 1]
]


vT = [
    [5],
    [10],
    [2],
    [6],
    [1]
]
v
m = [
    [8, 7, 1, 2, 3],
    [1, 5, 2, 9, 0],
    [8, 2, 2, 4, 1]
]

m[1][4] = 5

r = []
for i in range(len(m)):
    row = m[i]
    new_row = [] # empty row for now
    for j in range(len(row)):
        m_ij = m[i][j]
        r_ij = 5 * m_ij
        new_row.append(r_ij)
    r.append(new_row)
def matrix_print(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            m_ij = matrix[i][j]
            print(m_ij, '\t', end="")
        print('\n') # prints a new line
    return

m = [
    [8, 7, 1, 2, 3],
    [1, 5, 2, 9, 5],
    [8, 2, 2, 4, 1]
]

matrix_print(m)

assert v == [5, 10, 2, 6, 1]
assert mv == [
    [5, 10, 2, 6, 1]
]

assert vT == [
    [5], 
    [10], 
    [2], 
    [6], 
    [1]]

assert m == [
    [8, 7, 1, 2, 3], 
    [1, 5, 2, 9, 5], 
    [8, 2, 2, 4, 1]
]

assert r == [
    [40, 35, 5, 10, 15], 
    [5, 25, 10, 45, 25], 
    [40, 10, 10, 20, 5]
]
print(v)
print(mv)
print(vT)
print(m)
print(r)
