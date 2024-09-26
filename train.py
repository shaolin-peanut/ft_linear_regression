import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

import matplotlib
matplotlib.use('TkAgg')

def save_theta(theta):
    with open('theta', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(theta)

def estimate_price(mileage, theta):
    return theta[0] + (theta[1] * mileage)

def plot_cost(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.title('Loss over iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('cost_history.png')
    #plt.show(block=True)

def fit(x, y, alpha, iters):
    xlen = x.shape[0]
    theta = np.zeros(2)
    cost_history = []

    # save deviations to rescale later
    ystd = np.std(y)
    xstd = np.std(x)

    # scale it down
    x =  x / xstd 
    y = y / ystd

    for i in range(iters):
        y_pred = estimate_price(x, theta)
        residual = y_pred - y

        gradient_intercept = np.sum(residual) / xlen
        gradient_slope = np.sum(residual * x) / xlen

        theta[0] -= alpha * gradient_intercept
        theta[1] -= alpha * gradient_slope 

        cost = np.sum((y_pred - y) ** 2) / (2 * len(y))
        cost_history.append(cost_history)

        if i % 50 == 0:
            print(f"Iteration {i}: Intercept = {theta[0]:.4f}, Slope = {theta[1]:.4f}, Cost = {cost:.4f}")

    # scale it back up
    d_theta = np.zeros(2)
    d_theta[1] = theta[1] * np.std(y) / np.std(x)
    d_theta[0] = theta[0] * np.std(y)

    return d_theta, cost_history

def main():
    data = pd.read_csv("data.csv")

    x_train = data["km"].to_numpy()
    y_train = data["price"].to_numpy()

    print(x_train, y_train)

    theta, cost_history = fit(x_train, y_train, 0.1, 1000)

    save_theta(theta)

    #plot_cost(cost_history)


if __name__ == '__main__':
    main()
