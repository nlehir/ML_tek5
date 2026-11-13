"""
Used to batch over all the main files and several noise
standard deviations
"""

import subprocess
from pprint import pp

STDS = [0, 1, 10, 20]


def main():
    commands = []
    for std in STDS:
        command = ["uv", "run", "main_create_data.py", "--std", f"{std}"]
        commands.append(command)

        command = ["uv", "run", "main_random_params.py", "--std", f"{std}"]
        commands.append(command)

        command = ["uv", "run", "main_optimal_params_solution.py", "--std", f"{std}"]
        commands.append(command)

        command = ["uv", "run", "main_polynomial_regression.py", "--std", f"{std}"]
        commands.append(command)

    pp(commands)
    for command in commands:
        subprocess.run(command, check=False)


if __name__ == "__main__":
    main()
