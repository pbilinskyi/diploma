import numpy as np
import networkx as nx
from sympy import EX
import matplotlib.pyplot as plt
import os


test_example_num = int(input('# of data example (1 - 4): '))

r = np.loadtxt(os.path.join('data', f'image{test_example_num}_r.txt'))
c = np.loadtxt(os.path.join('data', f'image{test_example_num}_c.txt'))

image_height = r.shape[0]

c = c / c.sum().sum()
r = r / r.sum().sum()

r = r.ravel()
c = c.ravel()



# build cost matrix
n = len(r)

indexes = [(i, j) for i in range(image_height) for j in range(image_height)]

def distance(point1, point2):
    a, b = point2[0] - point1[0], point2[1] - point1[1]
    dist = np.hypot(a, b)
    # modification for dual approximation
    # dist = 0 if dist == 0 else 1/dist
    return dist


C = np.zeros(shape=(n, n))
for i_r in range(n):
    for i_c in range(n):
        d = distance(indexes[i_r], indexes[i_c])
        C[i_r, i_c] = d*d # squared distance



# l1 regression formulation
m = n*n

## initial approximation
def generate_initial_approximation():
    x_0 = np.array([1.0/m] * m)
    y_0 = np.array([0]*(2*n))
    return x_0, y_0

def x_to_matrix_plan(x):
    return np.vstack(np.split(x, n))

x_0, y_0 = generate_initial_approximation()

## incidence matrix
def generate_incidence_matrix_1():
    A = np.zeros(shape=(2*n, m))
    for i in range(n):
        for j in range(n):
            A[i, j*n:(j+1)*n] = 1
    return A


def generate_incidence_matrix():
    B = nx.Graph()
    nodes_from = np.arange(n)
    nodes_to = np.arange(n, 2*n)
    B.add_nodes_from(nodes_from, bipartite=0)
    B.add_nodes_from(nodes_to, bipartite=1)
    B.add_edges_from([(a, b) for a in nodes_from for b in nodes_to])
    A = nx.incidence_matrix(B)
    return A


def norm(x, type):
    if type == 'inf':
        return np.max(np.abs(x))
    elif type == '1':
        return np.sum(np.abs(x))
    elif type == '2':
        return np.sqrt(np.dot(x, x))
    else:
        raise Exception('Type of norm not specified')


b = np.hstack([r, c])
d = np.hstack(np.vsplit(C, n)).ravel()
norm_d = norm(d, type='inf')
A = generate_incidence_matrix()


# print(f'n = {n}')
# print(f'm = {m}')
# print(f'A = \n{A.toarray()}')
# print('d = \n{:}'.format(d))
# print('norm_d = {:.2e}'.format(norm_d))
# print(f'x_0 = \n{x_0}\ny_0 = \n{y_0}')