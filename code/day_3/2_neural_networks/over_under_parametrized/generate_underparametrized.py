"""
    genrate the dataset for an underparametrized problem
    (no estimator in the considered class interpolates the data)

    In this case adding a random noise to the output enforces this.

"""

import torch


def main() -> None:
    n_samples = 20
    dim_in = 10
    dim_h = 5
    dim_out = 1

    X = torch.randn(n_samples, dim_in)
    Neural_network = torch.nn.Sequential(
        torch.nn.Linear(dim_in, dim_h), torch.nn.ReLU(), torch.nn.Linear(dim_h, dim_out)
    )
    noise = torch.randn(n_samples, dim_out)
    y = Neural_network(X) + noise

    torch.save(X, "data/X_underparametrized")
    torch.save(y, "data/y_underparametrized")


if __name__ == "__main__":
    main()
