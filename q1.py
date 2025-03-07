import numpy as np
import matplotlib.pyplot as plt

from math import sin


def integrate_explicit_euler(f, x0, t_start, t_end, dt):
    xs = [x0]
    ts = [t_start]

    for t in np.arange(t_start + dt, t_end + dt, dt):
        x = xs[-1]
        x_new = x + f(x) * dt

        xs.append(x_new)
        ts.append(t)

    return np.array(xs), np.array(ts)


def integrate_implicit_euler(f, x0, t_start, t_end, dt):
    xs = [x0]
    ts = [t_start]

    eps = 1e-6

    for t in np.arange(t_start + dt, t_end + dt, dt):
        x = xs[-1]

        x_new = x

        while True:
            r_a = x + f(x_new - eps / 2) * dt - (x_new - eps / 2)
            r_b = x + f(x_new) * dt - x_new
            r_c = x + f(x_new + eps / 2) * dt - (x_new + eps / 2)

            dr = (r_c - r_a) / eps

            if np.all(np.isclose(r_b, 0)):
                break

            x_new = x_new - r_b / dr

        xs.append(x_new)
        ts.append(t)

    return np.array(xs), np.array(ts)


def integrate_explicit_rk4(f, x0, t_start, t_end, dt):
    xs = [x0]
    ts = [t_start]

    for t in np.arange(t_start + dt, t_end + dt, dt):
        x = xs[-1]

        k1 = f(x)
        k2 = f(x + (dt / 2) * k1)
        k3 = f(x + (dt / 2) * k2)
        k4 = f(x + dt * k3)

        x_new = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        xs.append(x_new)
        ts.append(t)


    return np.array(xs), np.array(ts)


def integrate_semiimplicit_euler(f, x0, t_start, t_end, dt):
    xs = [x0]
    ts = [t_start]

    for t in np.arange(t_start + dt, t_end + dt, dt):
        x = xs[-1]

        theta_dot_new = x[1] + f(x)[1] * dt
        theta_new = x[0] + theta_dot_new * dt

        x_new = np.array([theta_new, theta_dot_new])

        xs.append(x_new)
        ts.append(t)

    return np.array(xs), np.array(ts)


def integrate_implicit_midpoint(f, x0, t_start, t_end, dt):
    xs = [x0]
    ts = [t_start]

    eps = 1e-6

    for t in np.arange(t_start + dt, t_end + dt, dt):
        x = xs[-1]

        x_new = x

        while True:
            r_a = x + f((x + (x_new - eps / 2)) / 2) * dt - (x_new - eps / 2)
            r_b = x + f((x + x_new) / 2) * dt - x_new
            r_c = x + f((x + (x_new + eps / 2)) / 2) * dt - (x_new + eps / 2)

            dr = (r_c - r_a) / eps

            if np.all(np.isclose(r_b, 0)):
                break

            x_new = x_new - r_b / dr

        xs.append(x_new)
        ts.append(t)

    return np.array(xs), np.array(ts)


def mechanical_energy(xs, m, g, L, I):
    return (1 / 2) * (m * L ** 2 + I) * (xs[:, 1] ** 2) - m * g * L * np.cos(xs[:, 0])


def main():
    m = 1
    I = 1e-3
    g = 9.81
    L = g / (4 * np.pi ** 2)

    x0 = np.array([np.pi / 6, 0])

    dt = 1e-1

    t_start = 0
    t_end = 1000

    def f(x):
        return np.array([x[1], 1 / (m * L ** 2 + I) * (-m * g * L * sin(x[0]))])

    xs_explicit_euler, ts = integrate_explicit_euler(f, x0, t_start, t_end, dt)
    xs_implicit_euler, _ = integrate_implicit_euler(f, x0, t_start, t_end, dt)
    xs_explicit_rk4, _ = integrate_explicit_rk4(f, x0, t_start, t_end, dt)
    xs_semiimplicit_euler, _ = integrate_semiimplicit_euler(f, x0, t_start, t_end, dt)
    xs_implicit_midpoint, _ = integrate_implicit_midpoint(f, x0, t_start, t_end, dt)

    plt.plot(ts, xs_explicit_euler[:, 0], label="Explicit Euler")
    plt.plot(ts, xs_implicit_euler[:, 0], label="Implicit Euler")
    plt.plot(ts, xs_explicit_rk4[:, 0], label="Explicit RK4")
    plt.plot(ts, xs_semiimplicit_euler[:, 0], label="Semiimplicit Euler", ls="dashed", dashes=(5, 5))
    plt.plot(ts, xs_implicit_midpoint[:, 0], label="Implicit Midpoint", ls="dashed", dashes=(5, 10))
    plt.xlabel("Time")
    plt.ylabel(r'$\theta$')
    plt.title("Trajectory")
    plt.legend()
    plt.show()

    es_explicit_euler = mechanical_energy(xs_explicit_euler, m, g, L, I)
    es_implicit_euler = mechanical_energy(xs_implicit_euler, m, g, L, I)
    es_explicit_rk4 = mechanical_energy(xs_explicit_rk4, m, g, L, I)
    es_semiimplicit_euler = mechanical_energy(xs_semiimplicit_euler, m, g, L, I)
    es_implicit_midpoint = mechanical_energy(xs_implicit_midpoint, m, g, L, I)

    plt.plot(ts, es_explicit_euler, label="Explicit Euler")
    plt.plot(ts, es_implicit_euler, label="Implicit Euler")
    plt.plot(ts, es_explicit_rk4, label="Explicit RK4")
    plt.plot(ts, es_semiimplicit_euler, label="Semiimplicit Euler")
    plt.plot(ts, es_implicit_midpoint, label="Implicit Midpoint", ls="dashed", dashes=(5, 5))
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Mechanical Energy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()