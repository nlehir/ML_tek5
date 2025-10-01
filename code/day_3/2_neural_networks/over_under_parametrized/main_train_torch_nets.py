import os

import matplotlib.pyplot as plt
import torch

SETTING = "underparametrized"
# SETTING = "overparametrized"

# load data
X = torch.load(f"data/X_{SETTING}")
y = torch.load(f"data/y_{SETTING}")

n_samples, dim_in = X.shape
dim_out = y.shape[1]

# overparametrized
dim_h = 80
print(f"n_samples: {n_samples}")
print(f"dim_in: {dim_in}")
print(f"dim_h: {dim_h}")
print(f"dim_out: {dim_out}")

loss_fn = torch.nn.MSELoss(reduction="sum")
N_ITERATIONS = int(1e5)
NB_STORED_STEPS = 200
stored_iterations = [
    k * (N_ITERATIONS / NB_STORED_STEPS) for k in range(NB_STORED_STEPS)
]


def test_learning_rate(learning_rate: float) -> list[float]:
    """
    Train a neural network with one hidden layer
    by SGD with a constant learning rate.
    """
    print(f"\ntest learning rate {learning_rate:.3E}")

    # initialize neural network
    Neural_network = torch.nn.Sequential(
        torch.nn.Linear(dim_in, dim_h), torch.nn.ReLU(), torch.nn.Linear(dim_h, dim_out)
    )
    # initialize optimizer
    optim = torch.optim.SGD(Neural_network.parameters(), lr=learning_rate, momentum=0.9)

    # run SGD
    losses = list()
    for iteration in range(N_ITERATIONS):
        pred_y = Neural_network(X)
        loss = loss_fn(pred_y, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if iteration in stored_iterations:
            print(f"iteration: {iteration},  loss: {loss:.3E}")
            losses.append(loss.item())
    return losses


def main() -> None:
    # compare several learning rates
    learning_rates = [0.005] + [10 ** (-k) for k in range(3, 6)]
    for learning_rate in learning_rates:
        losses = test_learning_rate(learning_rate)
        plt.plot(
            stored_iterations,
            losses,
            "o",
            label=r"$\gamma=$" + f"{learning_rate}",
            markersize=3,
            alpha=0.6,
        )
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="best")
    title = (
        "learning curves: SGD, one hidden layer NN\n"
        + SETTING
        + f"\ninput dim: {dim_in}, n_samples size: {n_samples}"
        + f"\nhidden dim: {dim_h}"
        + f"\noutput dim: {dim_out}"
    )
    plt.title(title)
    plt.tight_layout()
    fig_path = os.path.join(
        "images",
        f"learning_rates_{SETTING}.pdf",
    )
    plt.savefig(fig_path)


if __name__ == "__main__":
    main()
