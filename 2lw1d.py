
import inline as inline
import matplotlib

import scipy
import itertools
import numpy as np
import matplotlib.pyplot as plt
import math


def swap_halves(a):
    n = len(a)
    assert n % 2 == 0

    m = n // 2

    b = a
    b = np.r_[
        b[m:],
        b[:m],
    ]

    return b


def left_right_pad(a, m):
    n = len(a)

    assert n % 2 == 0
    assert m % 2 == 0
    assert m >= n

    l = (m - n) // 2
    r = l + n

    b = np.zeros(m, dtype=a.dtype)
    b[l:r] = a

    return b


def left_right_unpad(b, n):
    m = len(b)

    assert n % 2 == 0
    assert m % 2 == 0
    assert m >= n

    l = (m - n) // 2
    r = l + n

    return b[l:r]


def ascomplex(a):
    return np.array(a, dtype=np.complex)


def finite_fft(n, a_f, step_f, ys_f_shifted, m):
    assert n % 2 == 0
    assert m % 2 == 0

    fft_arg = ys_f_shifted
    fft_arg = left_right_pad(fft_arg, m)
    fft_arg = swap_halves(fft_arg)

    fft_res = np.fft.fft(fft_arg)

    ys_F = fft_res * step_f
    ys_F = swap_halves(ys_F)
    ys_F = left_right_unpad(ys_F, n)

    return ys_F


def finite_integral(n, step_f, xs_f, ys_f, xs_F):
    # first dimension - x
    x_2d = np.broadcast_to(xs_f[:, np.newaxis], (n, n))

    # second dimension - u
    u_2d = np.broadcast_to(xs_F[np.newaxis, :], (n, n))

    # exp values
    A = np.exp((-2 * np.pi * 1j) * x_2d * u_2d)

    # scale rows by f(x)
    A = A * np.broadcast_to(ys_f[:, np.newaxis], (n, n))

    int_weights = np.ones(n)
    int_weights[0] = 1 / 2
    int_weights[-1] = 1 / 2
    int_weights *= step_f

    # scale rows by int_weights
    A = A * np.broadcast_to(int_weights[:, np.newaxis], (n, n))

    ys_F = np.sum(A, axis=0)

    return ys_F


def draw_1d(sp_n, sp_m, sp_c, xs, ys, s):
    plt.subplot(sp_n, sp_m, sp_c + 1)
    plt.plot(xs, np.abs(ys))
    plt.title(f'$\\left|{s}\\right|$')

    plt.subplot(sp_n, sp_m, sp_c + 2)
    plt.plot(xs, np.angle(ys))
    plt.ylim(-np.pi * 1.1, np.pi * 1.1)
    plt.title(f'$\\angle {s}$')

    plt.subplot(sp_n, sp_m, sp_c + 3)
    plt.plot(xs, np.real(ys))
    plt.title(f'$\\Re {s}$')

    plt.subplot(sp_n, sp_m, sp_c + 4)
    plt.plot(xs, np.imag(ys))
    plt.title(f'$\\Im {s}$')

# gauss

f = lambda x: np.exp(-x**2)
#F = lambda x: np.sqrt(np.pi) * np.exp(-x**2 * np.pi**2)

n = 1 << 10
m = 1 << 15

a_f = 5
# var Yam

n = 1 << 10
m = 1 << 15

a_f = 5

f = lambda x: x
#F = lambda x: 2 * ((np.pi * x * a_f) * np.sin(2 * (np.pi * x * a_f)) + 2 * (np.pi * x * a_f) * np.cos(2 * (np.pi * x * a_f)) - np.sin(2 * (np.pi * x * a_f)) / (2 * np.pi**2 * x**2))

# prep
a_F = n ** 2 / (4 * a_f * m)

step_f = 2 * a_f / (n - 1)
step_F = 2 * a_F / (n - 1)

xs_f = np.linspace(-a_f, a_f, n)
xs_f_shifted = xs_f - step_f / 2
xs_F = np.linspace(-a_F, a_F, n)

ys_f = ascomplex(f(xs_f))
ys_f_shifted = ascomplex(f(xs_f_shifted))

# analytical
#ys_F_analytical = ascomplex(F(xs_F))

# fft
ys_F_fft = ascomplex(finite_fft(n, a_f, step_f, ys_f_shifted, m))

# integral
ys_F_integral = ascomplex(finite_integral(n, step_f, xs_f, ys_f, xs_F))

fig, axs = plt.subplots()
draw_1d(4, 4, 0, xs_f, ys_f, 'f')

draw_1d(4, 4, 4, xs_F, ys_F_fft, 'F_{fft}')
draw_1d(4, 4, 8, xs_F, ys_F_integral, 'F_{integral}')
#draw_1d(4, 4, 12, xs_F, ys_F_analytical, 'F')
plt.show()