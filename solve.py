import numpy as np
import matplotlib.pyplot as plt
import os

from init import *

np.set_printoptions(precision=3, suppress=True)

max_iter = int(input('Maximal # of iterations: '))
eps = float(input('Accuracy (epsilon): '))
L = 2 * norm(d, type='inf') * 2

# objective function
def compute_objective(x, y):
    res = np.dot(d, x) + 2 * norm(d, type='inf') * (np.dot(y, A.dot(x)) - np.dot(b, y))
    return res


# PROXIMAL ALGORITHM

# bregmann diergence

def neg_entropy(x_1, x_2):
    res = 0
    for i in len(x_1):
        if x_1[i] == 0:
            res += 0
        else:
            res += x_1[i] * np.log(x_2[i])
    return res


def reg_x(x):
    return neg_entropy(x, x)


def reg_y(y):
    y_norm = norm(y, type='2')
    return 0.5 * y_norm * y_norm


#def D_x(x_1, x_2):
    return np.dot(x_1, np.ln(x_1 / x_2))


#def D_y(y_1, y_2):
    return 0.5 * np.square(norm(y_1 - y_2, type='2'))


# regularizers
#def r(x, y):
    return reg_x(x) + reg_y(y)


#def r_sidfort(x, y):
#    return 2 * norm_d * (10 * neg_entropy(x, x) + np.dot(x, A.transpose().dot(np.square(y))))


def project_hypercube(y):
    '''
        Projection of d-dimensional vector y onto hypercube [-1, 1]^d
    '''
    y_projected = np.zeros(len(y))
    for i in range(len(y)):
        if y[i] > 1:
            y_projected[i] = 1
        elif y[i] < -1:
            y_projected[i] = -1
        else:
            y_projected[i] = y[i]
    return y_projected
    # short form: y_projected[i] = 1 if y[i] > 1 else (-1 if y[i] < -1 else y[i])


def prox_x(x, a):
    '''
        Proximal operator on simplex based on Bregman's divergence and Cross entropy as regularizer
    '''
    res = x * np.exp(a)
    return res / res.sum()


def prox_y(y, a):
    return project_hypercube(y + a)


def prox(s, a):
    x, y = s[:m], s[m:]
    a_x, a_y = a[:m], a[m:]
    return np.concatenate((prox_x(x, a_x), prox_y(y, a_y)))


def g(x, y):
    '''
        Gradient operator for saddle-point problem
    '''
    norm_d = norm(d, type='inf')
    grad_x = d + 2 * norm_d * (A.transpose().dot(y))
    neg_grad_y = 2 * norm_d * (b - A.dot(x))
    return np.concatenate((grad_x, neg_grad_y))


def operator_A(s):
    x, y = s[:m], s[m:]
    return g(x, y)


# Korpelevich method with Bregmans divergence

s = np.concatenate((x_0, y_0))


def deviation(s, t):
    return norm(s - t, type='2')


def cost(x):
    return np.dot(d, x)


def conditions_penalty(x):
    return norm(A.dot(x) - b, type='1')


def objective(s):
    x, y = s[:m], s[m:]
    return cost(x) + 2 * norm(d, type='inf') * (np.dot(y, A.toarray().dot(x)) - np.dot(b, y))


def grad_x_r(x, y):
    return 2 * norm_d * (10 * (1 + np.log(x)) + A.transpose().dot(np.square(y)))


def grad_y_r(x, y):
    return 2 * norm_d * (2 * y * A.dot(x))


def f_prox(z, w, s_x, s_y):
    x, y = z[:m], z[m:]
    w_x, w_y = w[:m], w[m:]
    return np.dot(s_x, w_x) + np.dot(s_y, w_y) + r_sidfort(w_x, w_y) - r_sidfort(x, y) - (
            np.dot(grad_x_r(x, y), w_x - x) + np.dot(grad_y_r(x, y), w_y - y))


def update_s(x, y, s_x, s_y, coeff):
    s_x_new = s_x + coeff * (d + 2 * norm_d * A.transpose().dot(y))
    s_y_new = s_y + coeff * (2 * norm_d * (b - A.dot(x)))
    return s_x_new, s_y_new


def approximate_prox(x, y, s_x, s_y, eps=0.1, verbose=False):
    '''
        Primitive for computing approximate Prox_{z_t}(s_t) with regularizer r(x, y) from Sidfort's article
        
        Parameters
        ----------
        x : numpy.ndarray, m-dimensional
            An x-part of point z_t, relative to what proximation is made
        y : numpy.ndarray, 2*n-dimensional
            An y-part of point z_t, relative to what proximation is made
        s_x : numpy.ndarray, lm-dimensional
        s_y : numpy.ndarray, 2*n-dimensional

        Returns
        -------
        x, y - approximate solution of a constrained minimization problem Prox_{z_t}(s_t)
        
    '''
    # constant of 
    theta = 20 * norm_d * np.log(n) + 4 * norm_d
    n_iterations = int(np.ceil(
        24 * np.log(
            (
                    (88 * norm_d) / (eps * eps) + 2 / eps
            ) * theta
        )
    ))
    for k in range(n_iterations):
        temp = 1/(2*norm_d) * (-s_x) - 1/10 * A.transpose().toarray().dot(y * y)
        x = np.exp(temp)
        x = x / np.abs(x).sum()
        y = (-s_y ) / (4 * norm_d * A.toarray().dot(x))
        y[y < -1] = -1
        y[y > 1] = 1

    return x, y



def approximate_prox_s(z, s):
    x, y = z[:m], z[m:]
    s_x, s_y = s[:m], s[m:]
    # constant of 
    theta = 20 * norm_d * np.log(n) + 4 * norm_d
    n_iterations = int(np.ceil(
        24 * np.log(
            (
                    (88 * norm_d) / (eps * eps) + 2 / eps
            ) * theta
        )
    ))
    for k in range(n_iterations):
        temp = (-s_x) - (2*norm_d) * A.transpose().toarray().dot(y * y)
        # temp = (-s_x ) - (2*norm_d)*A.transpose().toarray().dot(y * y)
        x = np.exp(temp)
        x = x / np.abs(x).sum()
        y = (-s_y ) / (4 * norm_d * A.toarray().dot(x))
        y[y < -1] = -1
        y[y > 1] = 1
    return np.concatenate([x, y])



def wasserstein_distance(x):
    return np.sqrt(cost(x))


def duality_gap(x, y, verbose=False):
    if verbose:
        print('sum of x_i = {:.5f}'.format(x.sum()))
        print('max |y_i| = {:.5f}'.format(np.abs(y).max()))
    L_max_v = np.dot(d, x) + 2 * norm(d, type='inf') * norm(A.dot(x) - b, type='1')
    if verbose:
        print('''
L_max_v = {:.5f}
    d = {:}
    np.dot(d, x) = {:.5f}
    2 * norm(d, type='inf') = {:5f}
    norm(A.dot(x) - b, type='1') = {:.5f}
'''.format(L_max_v, d, np.dot(d, x), 2 * norm(d, type='inf'), norm(A.dot(x) - b, type='1')))
    
    L_min_u = -2 * norm_d * np.dot(b, y) + (d + 2 * norm_d * A.transpose().dot(y)).min()
    
    if verbose:
        print('''L_min_u = {:.5f}
    np.dot(b, y) = {:}
'''.format(L_min_u,  np.dot(b, y),    ))
        print('Duality gap: {:.5f}, max_y L(x, y) = {:.5f}, min_x L(x, y) = {:.5f}'.format(L_max_v - L_min_u, L_max_v, L_min_u))
    return L_max_v - L_min_u


def print_algo_info(t, x, y):
    print('''# {:}
W_eps = {:.10f}
    cost = {:.10f}
Duality gap = {:.10f}
||Ax - b|| = {:.10f}
'''.format(t, wasserstein_distance(x), cost(x), duality_gap(x, y), conditions_penalty(x)))


def korpelevich():
    print('> Korpelevich')
    lmbda = 1 / L
    i = 0
    x, y = generate_initial_approximation()
    s = np.concatenate((x, y))
    s_memory = [s]
    dgap = duality_gap(x, y)

    while dgap >= eps and i < max_iter:
        i += 1

        t = prox(s, -lmbda * operator_A(s))
        s = prox(s, -lmbda * operator_A(t))

        x, y = s[:m], s[m:]
        dgap = duality_gap(x, y)
        # print('dgap = {:.5f}'.format(dgap))
        s_memory.append(s)

    print_algo_info(i, x, y)
    return s, s_memory


def korpelevich_approx():
    print('> Korpelevich with r')
    lmbda = 1 / L
    i = 0
    x, y = generate_initial_approximation()
    s = np.concatenate((x, y))
    s_memory = [s]
    dgap = duality_gap(x, y)

    while dgap >= eps and i < max_iter:
        i += 1

        a = g(s[:m], s[m:])
        print('s_x = {:}'.format(s[:m]))
        print('s_y = {:}'.format(s[m:]))
        print('g_x = {:}'.format(a[:m]))
        print('prox')
        t = approximate_prox_s(s, lmbda * a)
        a = g(t[:m], t[m:])
        print('t_x = {:}'.format(t[:m]))
        print('t_y = {:}'.format(t[m:]))
        print('g_x = {:}'.format(a[:m]))
        print('prox')
        s = approximate_prox_s(s, lmbda * a)

        x, y = s[:m], s[m:]
        dgap = duality_gap(x, y)
        print('dgap = {:.5f}'.format(dgap))
        s_memory.append(s)

    print_algo_info(i, x, y)
    return s, s_memory


def tseng():
    print('> Tseng')
    lmbda = 1 / (5*L)
    i = 0
    x, y = generate_initial_approximation()
    s = np.concatenate((x, y))
    s_memory = [s]
    dgap = duality_gap(x, y)
    while dgap >= eps and i < max_iter:
        i += 1
 
        t = prox(s, -lmbda * operator_A(s))
        s = t - lmbda * (operator_A(t) - operator_A(s))

        x, y = s[:m], s[m:]
        dgap = duality_gap(x, y)

        s_memory.append(s)

    print_algo_info(i, x, y)
    return s, s_memory


def tseng_approx():
    print('> Tseng')
    lmbda = 1 / (5*L)
    i = 0
    x, y = generate_initial_approximation()
    s = np.concatenate((x, y))
    s_memory = [s]
    dgap = duality_gap(x, y)
    while dgap >= eps and i < max_iter:
        i += 1
 
        t = approximate_prox_s(s, -lmbda * operator_A(s))
        s = t - lmbda * (operator_A(t) - operator_A(s))

        x, y = s[:m], s[m:]
        dgap = duality_gap(x, y)

        s_memory.append(s)

    print_algo_info(i, x, y)
    return s, s_memory


def operator_extrapolation():
    print('> Operator Extrapolation')
    lmbda = 1 / (2 * L)
    lmbda_prev = lmbda
    i = 1
    # initial approximation
    x, y = generate_initial_approximation()
    s_prev = np.concatenate((x, y))
    s = prox(s_prev, -lmbda * operator_A(s_prev))

    s_memory = [s_prev, s]
    dgap = duality_gap(x, y)
    while dgap >= eps and i < max_iter:
        i += 1
        temp = s
        a = operator_A(s)
        s = prox(s, -lmbda * a - lmbda_prev * (a - operator_A(s_prev)))
        s_prev = temp

        x, y = s[:m], s[m:]
        dgap = duality_gap(x, y)
        s_memory.append(s)

    print_algo_info(i, x, y)
    return s, s_memory
    
    
# Sidford's algorithm

def sidford():
    print('> Sidford dual extrapolation')
    z_memory = []

    x = np.array([1.0/m] * m)  # [1/m, 1/m, ... , 1/m]
    y = np.array([0]*(2*n))  # [0, 0, ..., 0]
    s_x, s_y = np.zeros(m), np.zeros(2 * n)
    x_next_05, y_next_05 = x, y
    t = 0
    dgap = duality_gap(x_next_05, y_next_05)
    while dgap >= eps and t < max_iter:
        print('# {:}, duality gap = {:.5f}'.format(t, dgap))
        t += 1
        x_prev_05, y_prev_05 = x_next_05, y_next_05

        x, y = approximate_prox(x_prev_05, y_prev_05, s_x, s_y, verbose=False)
        s_x_next_05, s_y_next_05 = update_s(x, y, s_x, s_y, coeff=(1 / 3))
        x_next_05, y_next_05 = approximate_prox(x, y, s_x_next_05, s_y_next_05, verbose=False)
        s_x, s_y = update_s(x_next_05, y_next_05, s_x, s_y, coeff=(1 / 6))
        
        dgap = duality_gap(x_next_05, y_next_05)
        z_memory.append(np.concatenate((x_next_05, y_next_05)))
    print_algo_info(t, x, y)
    return np.concatenate((x_next_05, y_next_05)), z_memory


# TESTING 

def plot_objective_convergence(s_memory, label=''):
    import matplotlib.pyplot as plt
    iterations = np.arange(len(s_memory))
    objective_memory = np.array([objective(s) for s in s_memory])
    plt.plot(iterations, objective_memory, label=label)
    plt.legend()

def plot_duality_gap(s_memory, label, color='blue'):
    dg = [duality_gap(s[:m], s[m:]) for s in s_memory]
    plt.plot(np.arange(len(dg)), dg, color=color, label=label)

def vector_to_matrix(x, nrows):
    return np.vstack(np.split(x, nrows))

def save_history(algo_name, history):
    fname = os.path.join('result', f'{algo_name}_{test_example_num}.txt')
    np.savetxt(fname, np.array(history))

# MAIN

plt.figure(figsize=(10, 7))

s, s_memory = korpelevich()
save_history('korpelevich', s_memory)
plot_duality_gap(s_memory, label='Корпелевич', color='red')

s, s_memory = tseng()
save_history('tseng', s_memory)
plot_duality_gap(s_memory, label='Tseng', color='blue')

s, s_memory = operator_extrapolation()
save_history('operator_extrapolation', s_memory)
plot_duality_gap(s_memory, label='Операторної екстраполяції', color='green')

s, s_memory = sidford()
save_history('sidford', s_memory)
plot_duality_gap(s_memory, label='Sidford', color='magenta')

#s, s_memory = korpelevich_approx()
#plot_duality_gap(s_memory, label='Корпелевич', color='red')
#s, s_memory = tseng_approx()
#plot_duality_gap(s_memory, label='Tseng approx', color='red')

plt.legend()
ax = plt.gca(); ax.set_xlabel("# of iteration"); ax.set_ylabel("Duality Gap"); plt.title('Convergence')
plt.savefig(os.path.join('result', 'duality_gap_convergence.png'))
plt.show()