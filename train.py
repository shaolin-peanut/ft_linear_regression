import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import argparse

def save_theta(theta):
    with open('theta', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(theta)

def plot_line(X, y, theta):
    plt.scatter(X, y)
    y_pred = estimate_price(X, theta)
    plt.plot(X, y_pred, color='red')
    plt.show()

def estimate_price(X, theta):
    return theta[0] + theta[1] * X

def fit_with_animation(X, y, alpha, iters):
    X_len = X.shape[0]
    theta = np.zeros(2)
    cost_history = []

    # scaling
    X_std = np.std(X)
    y_std = np.std(y)
    X = X / X_std
    y = y / y_std

    fig, ax = plt.subplots()
    ax.scatter(X, y, color="blue")

    line, = ax.plot([], [], color="red")
    plt.xlim(min(X), max(X))
    plt.ylim(min(y) - 0.1, max(y) + 0.1)

    iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    theta_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    cost_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)

    # animation update
    def linear_regression(i):
        y_pred = estimate_price(X, theta)
        residual = y_pred - y

        gradient_intercept = np.sum(residual) / X_len
        gradient_slope = np.sum(residual * X) / X_len

        cost = np.sum((y_pred - y) ** 2) / (2 * X_len)

        theta[0] -= alpha * gradient_intercept
        theta[1] -= alpha * gradient_slope

        line.set_data(X, estimate_price(X, theta))

        iteration_text.set_text(f"Iteration: {i + 1}")
        theta_text.set_text(f"theta0 = {theta[0]:.4f}, theta1 = {theta[1]:.4f}")
        cost_text.set_text(f"Cost = {cost:.4f}")
        return line, iteration_text, theta_text, cost_text

    # so basically the regression runs as a function of the animation update stuff
    ani = FuncAnimation(fig, linear_regression, frames=iters, repeat=False, interval=50)

    plt.show()
    return theta, cost_history

    # scale it back
    theta[0] *= ystd
    theta[1] = (theta[1] * ystd) / xstd

    return theta, cost_history

def fit(x, y, alpha, iters):
    m = x.shape[0]
    theta = np.zeros(2)
    cost_history = []

    xstd = np.std(x)
    ystd = np.std(y)

    x =  x / xstd 
    y = y / ystd

    for i in range(iters):
        y_pred = estimate_price(x, theta)
        residual = y_pred - y

        gradient_intercept = np.sum(residual) / m
        gradient_slope = np.sum(residual * x) / m

        theta[0] -= alpha * gradient_intercept
        theta[1] -= alpha * gradient_slope

        cost = np.sum((y_pred - y) ** 2) / (2 * m)
        cost_history.append(cost)

        if i % 50 == 0:
            print(f"Iteration {i}: Intercept = {theta[0]:.4f}, Slope = {theta[1]:.4f}, Cost = {cost:.4f}")

    # scale it back
    theta[0] *= ystd
    theta[1] = (theta[1] * ystd) / xstd

    return theta, cost_history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--animation', action='store_true', help='Enable animation during fitting')
    args = parser.parse_args()

    data = pd.read_csv("data.csv")

    x_train = data["km"].to_numpy()
    y_train = data["price"].to_numpy()

    print(x_train, y_train)

    if args.animation:
        theta, _ = fit_with_animation(x_train, y_train, 0.1, 1000)
    else:
        theta, _ = fit(x_train, y_train, 0.1, 1000)
    
    save_theta(theta)

    #plot_cost(cost_history)


if __name__ == '__main__':
    main()
