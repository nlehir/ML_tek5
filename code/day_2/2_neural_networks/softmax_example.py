import torch

"""
Les 10 sorties du réseau
"""
z_vector = torch.tensor(
    [
        12,
        10,
        15,
        1,
        3,
        4,
        16,
        1,
        5,
        10,
    ]
)

"""
Conversion en softmax
"""
exp = torch.exp(z_vector)
softmax = exp / exp.sum()
print(f"{softmax=}")
__import__("ipdb").set_trace()
