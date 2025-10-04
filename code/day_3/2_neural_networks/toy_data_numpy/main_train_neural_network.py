"""
In this script a neural network tries to fit some manually generated data
"""
# import plot_net
import os

import matplotlib.pyplot as plt
import numpy as np
from utils import mean_squared_error, nn_r2
from sklearn.metrics import r2_score

# Hyperparameters
HIDDEN_DIM = 3
LEARNING_RATE = 1e-8
# plotting constants
NB_ITERATIONS = 50000
NB_PLOTTED_ITERATIONS = 100
# select the data
NOISE_STD_DEV = 1


def train_neural_net(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Perform empirical risk minimization by gradient descent
    and backpropagation on a neural network with one hidden layer.
    """
    # get problem dimensions
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]


    # Randomly initialize weights
    rng = np.random.default_rng()
    w1 = rng.normal(loc=0, scale=1, size=(input_dim, HIDDEN_DIM))
    w2 = rng.normal(loc=0, scale=1, size=(HIDDEN_DIM, output_dim))

    # we will store the loss as a function of the
    # iteration
    MSE_train_list = list()
    R2_train_list = list()
    MSE_test_list = list()
    R2_test_list = list()
    iterations = list()
    for iteration in range(NB_ITERATIONS):
        # forward pass
        # Prediction of the network for a given input x
        hh_train = x_train @ w1
        h_relu_train = np.maximum(hh_train, 0)
        y_pred_train = h_relu_train @ w2

        hh_test = x_test @ w1
        h_relu_test = np.maximum(hh_test, 0)
        y_pred_test = h_relu_test @ w2

        # loss function
        # Here we use the mean squared error (MSE) loss
        MSE_train = np.square(y_pred_train - y_train).sum() / len(y_train)
        R2_train = r2_score(y_true=y_train, y_pred=y_pred_train)
        MSE_test = np.square(y_pred_test - y_test).sum() / len(y_test)
        R2_test = r2_score(y_true=y_test, y_pred=y_pred_test)

        # ----------------------
        # Plot the network and the loss
        if iteration % (NB_ITERATIONS / NB_PLOTTED_ITERATIONS) == 0:
            # print("plot net")
            print(f"iteration: {iteration}, train MSE: {MSE_train:.2E}")

            # keep storing the MSE loss and the iterations
            MSE_train_list.append(MSE_train)
            R2_train_list.append(R2_train)
            MSE_test_list.append(MSE_test)
            R2_test_list.append(R2_test)
            iterations.append(iteration)

            plt.plot(
                    R2_train_list,
                    "o",
                    alpha=0.5,
                    color="blue",
                    )
            plt.plot(
                    R2_test_list,
                    "o",
                    alpha=0.5,
                    color="orange",
                    )
            # plt.yscale("log")

            # set the limits of the plots
            title = (
                "R2\n"
                f"{HIDDEN_DIM} hidden neurons\n"
                f"noise std: {NOISE_STD_DEV:.3f}\n"
                f"train R2: {R2_train_list[-1]:.3f}\n"
                f"test R2: {R2_test_list[-1]:.3f}\n"
                f"learning rate: {LEARNING_RATE}"
                    )
            plt.title(title)
            plt.xlabel("iteration")
            plt.ylabel("R2")
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)

        # backpropagation
        # computation of the gradients of the loss function
        # with respect to w1 and w2
        grad_y_pred_train = 2.0 * (y_pred_train - y_train)
        grad_w2 = h_relu_train.T.dot(grad_y_pred_train)
        grad_h_relu_train = grad_y_pred_train.dot(w2.T)
        grad_h = grad_h_relu_train.copy()
        grad_h[hh_train < 0] = 0
        grad_w1 = x_train.T.dot(grad_h)

        # update the weigths
        w1 -= LEARNING_RATE * grad_w1
        w2 -= LEARNING_RATE * grad_w2

    # save the plot of the loss function
    final_train_R2 = nn_r2(
            x=x_train,
            y=y_train,
            w1=w1,
            w2=w2,
            )
    final_test_R2 = nn_r2(
            x=x_test,
            y=y_test,
            w1=w1,
            w2=w2,
            )
    plt.close()

    """
    Plot R2 learning curve
    """
    plt.plot(iterations, R2_train_list, label="train")
    plt.plot(iterations, R2_test_list, label="test")
    title = (
        f"R2\n"
        f"{HIDDEN_DIM} hidden neurons\n"
        f"noise std: {NOISE_STD_DEV:.3f}\n"
        f"Final train R2 {final_train_R2:.3f}\n"
        f"Final test R2 {final_test_R2:.3f}\n"
        f"learning rate: {LEARNING_RATE}"
            )
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel("Train R2")
    # plt.yscale("log")
    plt.legend(loc="best")
    plt.tight_layout()
    figname = f"r2_fun_d_in_{input_dim}_d_out_{output_dim}_hidden_{HIDDEN_DIM}_std_{NOISE_STD_DEV}_rate_{LEARNING_RATE}.pdf"
    fig_path = os.path.join("images", figname)
    plt.savefig(fig_path)
    plt.show()
    plt.close("all")


    """
    Plot MSE learning curve
    """
    plt.plot(iterations, MSE_train_list, label="train")
    plt.plot(iterations, MSE_test_list, label="test")
    title = (
        f"Mean squared error\n"
        f"{HIDDEN_DIM} hidden neurons\n"
        f"noise std: {NOISE_STD_DEV:.3f}\n"
        f"learning rate: {LEARNING_RATE}"
            )
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel("Mean squared error")
    plt.yscale("log")
    plt.legend(loc="best")
    plt.tight_layout()
    figname = f"mse_fun_d_in_{input_dim}_d_out_{output_dim}_hidden_{HIDDEN_DIM}_std_{NOISE_STD_DEV}_rate_{LEARNING_RATE}.pdf"
    fig_path = os.path.join("images", figname)
    plt.savefig(fig_path)
    plt.show()
    plt.close("all")

    print("----")
    print(f"train MSE : {mean_squared_error(x_test, y_test, w1, w2)}")
    print(f"test MSE : {mean_squared_error(x_test, y_test, w1, w2)}")
    print(f"train R2: {final_train_R2:.3f}")
    print(f"test R2: {final_test_R2:.3f}")


def main() -> None:
    # select the data
    folder = "data"
    x_train = np.load(os.path.join(folder, f"training_inputs_std_{NOISE_STD_DEV}.npy"))
    y_train = np.load(os.path.join(folder, f"training_outputs_std_{NOISE_STD_DEV}.npy"))
    x_test = np.load(os.path.join(folder, f"test_inputs_std_{NOISE_STD_DEV}.npy"))
    y_test = np.load(os.path.join(folder, f"test_outputs_std_{NOISE_STD_DEV}.npy"))

    # train
    train_neural_net(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


if __name__ == "__main__":
    main()
