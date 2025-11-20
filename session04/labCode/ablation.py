import matplotlib.pyplot as plt
import lab04_student
import numpy as np
import random
import torch
import json
import os

currentDirectory = os.path.dirname(os.path.abspath(__file__))


def ablationSkipConnections(addNew: bool, createGraph: bool) -> None:
    """
    Function to run ablation study on different skip connection variants.

    Saves results in JSON files in the ablation/skipConnections directory.

    Args:
        - addNew (bool): Whether to add new experiments or not.
        - createGraph (bool): Whether to create graphs from the results.

    Returns:
        - None
    """
    dataFolder = os.path.join(currentDirectory, "ablation", "skipConnections")

    # Make sure the data folder exists
    os.makedirs(dataFolder, exist_ok=True)

    files = os.listdir(dataFolder)

    variants = ["concat", "add", "attention", "none"]

    config = {
        "batch_size": 16,
        "learning_rate": 0.001,
        "epochs": 50,
        "image_size": 128,
        "storeData": False,
    }

    if addNew:
        for v in variants:
            # Get a seed
            seed = random.randint(100, 999)

            while f"{v}_{seed}.json" in files:
                seed = random.randint(100, 999)

            # Set seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            data = lab04_student.main(skip_mode=v, **config)

            # Save data to json
            with open(os.path.join(dataFolder, f"{v}_{seed}.json"), "w") as outfile:
                json.dump(data, outfile)

            # Reset the seed
            random.seed()
            np.random.seed()
            torch.manual_seed(torch.initial_seed())

    if createGraph:
        data = {v: [] for v in variants}

        for file in files:
            variant = file.split("_")[0]

            if variant in variants:
                with open(os.path.join(dataFolder, file), "r") as infile:
                    runData = json.load(infile)

                data[variant].append(runData)

        # Create a box plot for training_time
        plt.figure(figsize=(10, 6))
        training_times = [[run["training_time"] for run in data[v]] for v in variants]
        plt.boxplot(training_times, labels=variants)
        plt.title("Training Time by Skip Connection Variant")
        plt.ylabel("Training Time (seconds)")

        # Create a box plot for avg_gradient
        plt.figure(figsize=(10, 6))
        avg_gradients = [[run["avg_gradient"] for run in data[v]] for v in variants]
        plt.boxplot(avg_gradients, labels=variants)
        plt.title("Average Gradient by Skip Connection Variant")
        plt.ylabel("Average Gradient")

        # Create a box plot for memory_usage
        plt.figure(figsize=(10, 6))
        memory_usages = [[run["memory_usage"] for run in data[v]] for v in variants]
        plt.boxplot(memory_usages, labels=variants)
        plt.title("Memory Usage by Skip Connection Variant")
        plt.ylabel("Memory Usage (MB)")


def main(addNew: bool = False, createGraph: bool = True) -> None:
    """
    Main function to run ablation studies.

    Args:
        - addNew (bool): Whether to add new experiments or not.
        - createGraph (bool): Whether to create graphs from the results.

    Returns:
        - None
    """
    ablationSkipConnections(addNew, createGraph)


if __name__ == "__main__":
    while True:
        main(addNew=True, createGraph=False)
