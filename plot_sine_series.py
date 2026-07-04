"""
Plot three sine-based functions:

1. y = sin(x)
2. y = sin(x) + sin(x^2) / x^2
3. y = sin(x) + sum_{n=2}^{20} sin(x^n) / x^(n-1)
"""

import numpy as np
import matplotlib.pyplot as plt


def sin_power_over_same_power(x, power):
    """Compute sin(x^power) / x^power, using the limit 1 at x = 0."""
    denominator = x**power
    numerator = np.sin(denominator)
    return np.divide(
        numerator,
        denominator,
        out=np.ones_like(x, dtype=float),
        where=denominator != 0,
    )


def sin_power_over_lower_power(x, power):
    """Compute sin(x^power) / x^(power-1), using the limit 0 at x = 0."""
    numerator = np.sin(x**power)
    denominator = x ** (power - 1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(x, dtype=float),
        where=denominator != 0,
    )


def build_functions(x):
    """Return the three requested y-series."""
    y1 = np.sin(x)
    y2 = y1 + sin_power_over_same_power(x, 2)
    y3 = y1.copy()
    for power in range(2, 6):
        y3 += sin_power_over_lower_power(x, power)
    return y1, y2, y3


def main():
    x = np.linspace(-4, 4, 4000)
    y1, y2, y3 = build_functions(x)

    plt.figure(figsize=(12, 7))
    plt.plot(x, y1, label=r"$y=\sin(x)$", linewidth=2)
    plt.plot(x, y2, label=r"$y=\sin(x)+\frac{\sin(x^2)}{x^2}$", linewidth=2)
    plt.plot(
        x,
        y3,
        label=r"$y=\sin(x)+\sum_{n=2}^{20}\frac{\sin(x^n)}{x^{n-1}}$",
        linewidth=2,
    )

    plt.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    plt.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    plt.title("Sine Series Functions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sine_series_functions.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
