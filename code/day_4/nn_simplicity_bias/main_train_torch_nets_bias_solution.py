import os

import matplotlib.pyplot as plt
import torch
from utils import load_data, clean_filename
from constants import SIGMA


# load data
DATA_FOLDER = os.path.join("data")
X_train, X_test, y_train, y_test = load_data(folder=DATA_FOLDER)

LEARNING_RATES = [0.005] + [10 ** (-k) for k in range(3, 4)]
LEARNING_RATES = [10 ** (-k) for k in range(3, 7)]
DIM_H = [10, 20, 30, 40, 100, 200]

X_train = X_train.reshape(len(X_train), 1)
y_train = y_train.reshape(len(y_train), 1)
X_test = X_test.reshape(len(X_test), 1)
y_test = y_test.reshape(len(y_test), 1)
X_train = torch.tensor(X_train).type(torch.float)
X_test = torch.tensor(X_test).type(torch.float)
y_train = torch.tensor(y_train).type(torch.float)
y_test = torch.tensor(y_test).type(torch.float)
n_train_samples, dim_in = X_train.shape
dim_out = y_train.shape[1]

# overparametrized
print(f"n_train_samples: {n_train_samples}")
print(f"dim_in: {dim_in}")
print(f"dim_out: {dim_out}")

# loss_fn = torch.nn.MSELoss(reduction="sum")
loss_fn = torch.nn.MSELoss()
N_ITERATIONS = int(1e5)
NB_STORED_STEPS = 200
stored_iterations = [
    k * (N_ITERATIONS / NB_STORED_STEPS) for k in range(NB_STORED_STEPS)
]


def train_net(
    dim_h: int,
    learning_rate: float,
    X_train,
    X_test,
    y_train,
    y_test,
) -> dict:
    """
    Train a neural network with one hidden layer
    by SGD with a constant learning rate.
    """
    print(f"\nlearning rate {learning_rate:.3E}")

    # initialize neural network
    Neural_network = torch.nn.Sequential(
        torch.nn.Linear(dim_in, dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h, dim_out),
        # torch.nn.ReLU(),
    )
    # initialize optimizer
    optim = torch.optim.SGD(Neural_network.parameters(), lr=learning_rate, momentum=0.9)

    # run SGD
    train_losses = list()
    test_losses = list()
    for iteration in range(N_ITERATIONS):
        y_pred_train = Neural_network(X_train)
        train_loss = loss_fn(y_pred_train, y_train)
        optim.zero_grad()
        train_loss.backward()
        optim.step()
        y_pred_test = Neural_network(X_test)
        test_loss = loss_fn(y_pred_test, y_test)
        if iteration in stored_iterations:
            if iteration % 1000 == 0:
                print(f"{iteration=}/{N_ITERATIONS}")
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())

    """"""
    # X_test_plotted = X_test.sort(dim=0).values
    # # X_test_plotted = X_test
    # y_pred_test = Neural_network(X_test_plotted)
    # plt.plot(X_test_plotted.detach().cpu(), y_pred_test.detach().cpu())
    # fig_name = (
    #         f"pred_dim_h_{dim_h}_lr_{learning_rate:.4f}"
    #         )
    # fig_name = f"{clean_filename(fig_name)}.pdf"
    # fig_path = os.path.join(
    #     "images",
    #     fig_name,
    # )
    # plt.savefig(fig_path)
    # plt.close()
    dico = dict(
        train_losses=train_losses,
        test_losses=test_losses,
        neural_net=Neural_network,
    )
    return dico


def main() -> None:
    # compare several learning rates
    # learning_rates = [0.005] + [10 ** (-k) for k in range(3, 6)]
    for dim_h in DIM_H:
        study_dim(
            dim_h=dim_h,
            learning_rates=LEARNING_RATES,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )


def study_dim(
    dim_h,
    learning_rates,
    X_train,
    X_test,
    y_train,
    y_test,
):
    print(f"\ndim_h: {dim_h}")
    dico = dict()
    for learning_rate in learning_rates:
        dico[learning_rate] = train_net(
            dim_h=dim_h,
            learning_rate=learning_rate,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
    for learning_rate in learning_rates:
        train_losses = dico[learning_rate]["train_losses"]
        test_losses = dico[learning_rate]["test_losses"]
        plt.plot(
            stored_iterations,
            train_losses,
            "o",
            label=r"$\gamma=$" + f"{learning_rate}, train",
            markersize=3,
            alpha=0.6,
        )
        plt.plot(
            stored_iterations,
            test_losses,
            "o",
            label=r"$\gamma=$" + f"{learning_rate}, test",
            markersize=3,
            alpha=0.6,
        )
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="best", fontsize=7)
    title = (
        "learning curves: SGD, one hidden layer NN"
        + f"\ninput dim: {dim_in}"
        + f", output dim: {dim_out}"
        + f", n train samples: {n_train_samples}"
        + f"\nhidden dim: {dim_h}\n"
    )
    plt.title(title)
    plt.tight_layout()
    fig_path = os.path.join(
        "images",
        f"learning_rates_dim_h_{dim_h}.pdf",
    )
    plt.savefig(fig_path)
    plt.close()

    """
        Plot predictions
        """
    for learning_rate in learning_rates:
        Neural_network = dico[learning_rate]["neural_net"]
        X_test_plotted = X_test.sort(dim=0).values
        y_pred_plot = Neural_network(X_test_plotted)

        """
            Measure overfitting
            """
        y_pred_train = Neural_network(X_train)
        y_pred_test = Neural_network(X_test)
        train_loss = loss_fn(y_pred_train, y_train)
        test_loss = loss_fn(y_pred_test, y_test)

        """
            Plot
            """
        plt.plot(X_train, y_train, "o", label="train", alpha=0.8)
        plt.plot(X_test, y_test, "o", label="test", alpha=0.8)
        plt.plot(
            X_test_plotted.detach().cpu(),
            y_pred_plot.detach().cpu(),
            label="neural network",
        )
        plt.legend(loc="best")
        title = (
            "Predictions, one hidden layer NN\n"
            r"$\sigma=$"
            + f"{SIGMA}"
            + f"\nlearning rate: {learning_rate}"
            + f"\ninput dim: {dim_in}"
            + f", output dim: {dim_out}"
            + f", n train samples: {n_train_samples}"
            + f"\nhidden dim: {dim_h}"
            + f"\ntrain loss: {train_loss:.2E}, test loss: {test_loss:.2E},"
        )
        plt.title(title)
        plt.tight_layout()
        fig_name = f"pred_dim_h_{dim_h}_lr_{learning_rate:.2E}"
        fig_name = f"{clean_filename(fig_name)}.pdf"
        fig_path = os.path.join(
            "images",
            fig_name,
        )
        plt.savefig(fig_path)
        plt.close()


if __name__ == "__main__":
    main()
