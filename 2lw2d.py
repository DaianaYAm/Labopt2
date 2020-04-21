import itertools

import numpy as np
import matplotlib.pyplot as plt

def swap_halves_2d(a):
    n = len(a)
    assert n % 2 == 0

    m = n // 2

    b = a
    b = np.r_[
        b[m:, :],
        b[:m, :],
    ]
    b = np.c_[
        b[:, m:],
        b[:, :m],
    ]

    return b


def left_right_pad_2d(a, m):
    n = len(a)

    assert n % 2 == 0
    assert m % 2 == 0
    assert m >= n

    l = (m - n) // 2
    r = l + n

    b = np.zeros((m, m), dtype=a.dtype)
    b[l:r, l:r] = a

    return b


def left_right_unpad_2d(b, n):
    m = len(b)

    assert n % 2 == 0
    assert m % 2 == 0
    assert m >= n

    l = (m - n) // 2
    r = l + n

    return b[l:r, l:r]


def finite_fft_2d(n, a_f, step_f, ys_f_shifted, m):
    assert n % 2 == 0
    assert m % 2 == 0

    fft_arg = ys_f_shifted
    fft_arg = left_right_pad_2d(fft_arg, m)
    fft_arg = swap_halves_2d(fft_arg)

    fft_res = np.fft.fft2(fft_arg)
    ys_F = fft_res * step_f ** 2
    ys_F = swap_halves_2d(ys_F)
    ys_F = left_right_unpad_2d(ys_F, n)

    return ys_F


def finite_integral_2d(n, step_f, xs_f, ys_f, xs_F):
    shape = (n, n, n, n)

    # first dimension - x
    x_4d = np.broadcast_to(xs_f[:, np.newaxis, np.newaxis, np.newaxis], shape)
    # second dimension - y
    y_4d = np.broadcast_to(xs_f[np.newaxis, :, np.newaxis, np.newaxis], shape)

    # third dimension - u
    u_4d = np.broadcast_to(xs_F[np.newaxis, np.newaxis, :, np.newaxis], shape)
    # forth dimension - v
    v_4d = np.broadcast_to(xs_F[np.newaxis, np.newaxis, np.newaxis, :], shape)

    # exp values
    A = np.exp((-2 * np.pi * 1j) * (x_4d * u_4d + y_4d * v_4d))

    # scale d1 and d2 by f(x, y)
    A = A * np.broadcast_to(ys_f[:, :, np.newaxis, np.newaxis], shape)

    int_weights = np.ones(n)
    int_weights[0] = 1 / 2
    int_weights[-1] = 1 / 2
    int_weights *= step_f

    # scale d1 by int_weights
    A = A * np.broadcast_to(int_weights[:, np.newaxis, np.newaxis, np.newaxis], shape)
    # scale d2 by int_weights
    A = A * np.broadcast_to(int_weights[np.newaxis, :, np.newaxis, np.newaxis], shape)

    ys_F = A
    ys_F = np.sum(ys_F, axis=0)
    ys_F = np.sum(ys_F, axis=0)

    return ys_F


def draw_2d(sp_n, sp_m, sp_c, xs, ys, s):
    extent = [xs[0], xs[-1], xs[0], xs[-1]]

    plt.subplot(sp_n, sp_m, sp_c + 1)
    plt.imshow(np.abs(ys), extent=extent)
    plt.colorbar()
    plt.title(f'$\\left|{s}\\right|$')

    plt.subplot(sp_n, sp_m, sp_c + 2)
    plt.imshow(np.angle(ys), extent=extent, vmin=-np.pi, vmax=np.pi)
    plt.colorbar()
    plt.title(f'$\\angle {s}$')

    plt.subplot(sp_n, sp_m, sp_c + 3)
    plt.imshow(np.real(ys), extent=extent)
    plt.colorbar()
    plt.title(f'$\\Re {s}$')

    plt.subplot(sp_n, sp_m, sp_c + 4)
    plt.imshow(np.imag(ys), extent=extent)
    plt.colorbar()
    plt.title(f'$\\Im {s}$')

def ascomplex(a):
    return np.array(a, dtype=np.complex)

# gauss

n = 1 << 6
m = 1 << 8

a_f = 5

f_2d = lambda a: np.exp(-(a[:, :, 0]**2 + a[:, :, 1]**2))
#F_2d = lambda a: np.pi * np.exp(-(a[:, :, 0]**2 + a[:, :, 1]**2) * np.pi**2)
# var 1

n = 1 << 6
m = 1 << 8

a_f = 5

f_2d = lambda a: np.sinc(a[:, :, 0]) * np.sinc(a[:, :, 1])
#F_2d = lambda a: (np.abs(a[:, :, 0]) <= 1/2) * 1 * (np.abs(a[:, :, 1]) <= 1/2) * 1

# var Yam

n = 1 << 6
m = 1 << 8

a_f = 5

f_2d = lambda a: a[:, :, 0]  * a[:, :, 1]
#F_2d = lambda a: (
 #   (
 #      2 * (np.pi * a[:, :, 0] * a_f) * np.sin(2 * (np.pi * a[:, :, 0] * a_f)) +
 #    2 * (np.pi * a[:, :, 0] * a_f) * np.cos(2 * (np.pi * a[:, :, 0] * a_f)) -
 #       np.sin(2 * (np.pi * a[:, :, 0] * a_f))
 #   ) / (2 * np.pi**2 * a[:, :, 0]**2)
 #   *
 #   (
 #       2 * (np.pi * a[:, :, 1] * a_f) * np.sin(2 * (np.pi * a[:, :, 1] * a_f)) +
 #       2 * (np.pi * a[:, :, 1] * a_f) * np.cos(2 * (np.pi * a[:, :, 1] * a_f)) -
 #       np.sin(2 * (np.pi * a[:, :, 1] * a_f))
 #   ) / (2 * np.pi**2 * a[:, :, 1]**2)
#)

# prep
a_F = n ** 2 / (4 * a_f * m)

step_f = 2 * a_f / (n - 1)
step_F = 2 * a_F / (n - 1)

xs_f = np.linspace(-a_f, a_f, n)
xs_f_shifted = xs_f - step_f / 2
xs_F = np.linspace(-a_F, a_F, n)

Xs_f = np.reshape(list(itertools.product(xs_f, xs_f)), (n, n, 2))
Xs_f_shifted = np.reshape(list(itertools.product(xs_f_shifted, xs_f_shifted)), (n, n, 2))
Xs_F = np.reshape(list(itertools.product(xs_F, xs_F)), (n, n, 2))

ys_f = ascomplex(f_2d(Xs_f))
ys_f_shifted = ascomplex(f_2d(Xs_f_shifted))

# analytical
#ys_F_analytical = ascomplex(F_2d(Xs_F))

# fft
ys_F_fft = ascomplex(finite_fft_2d(n, a_f, step_f, ys_f_shifted, m))

# integral
ys_F_integral = ascomplex(finite_integral_2d(n, step_f, xs_f, ys_f, xs_F))

fig, axs = plt.subplots()
draw_2d(4, 4, 0, xs_f, ys_f, 'f')
draw_2d(4, 4, 4, xs_F, ys_F_fft, 'F_{fft}')
draw_2d(4, 4, 8, xs_F, ys_F_integral, 'F_{integral}')
#draw_2d(4, 4, 12, xs_F, ys_F_analytical, 'F')
plt.show()