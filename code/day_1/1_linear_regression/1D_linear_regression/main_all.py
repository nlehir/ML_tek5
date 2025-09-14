"""
Used to batch over all the main files and several noise
standard deviations
"""

import os
from pprint import pp

STDS = [0, 1, 10, 20]

def main():
    commands = list()
    for std in STDS:
        command = (
                "uv run create_data.py "
                f" --std {std}"
                )
        commands.append(command)

        command = (
                "uv run main_random_params.py "
                f" --std {std}"
                )
        commands.append(command)

        command = (
                "uv run main_optimal_params_solution.py "
                f" --std {std}"
                )
        commands.append(command)

        command = (
                "uv run main_polynomial_regression.py "
                f" --std {std}"
                )
        commands.append(command)

    pp(commands)
    for command in commands:
        os.system(command)

if __name__ == "__main__":
    main()
