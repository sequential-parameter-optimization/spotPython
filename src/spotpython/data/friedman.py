import math
import random


class FriedmanDriftDataset:
    """Friedman Drift Dataset."""

    def __init__(self, n_samples=100, change_point1=50, change_point2=75, seed=None, constant=False) -> None:
        """Constructor for the Friedman Drift Dataset.

        Args:
            n_samples (int): The number of samples to generate.
            change_point1 (int): The index of the first change point.
            change_point2 (int): The index of the second change point.
            seed (int): The seed for the random number generator.
            constant (bool): If True, only the first feature is set to 1 and all others are set to 0.

        Returns:
            None (None): None

        Examples:
            >>> from spotpython.data.friedman import FriedmanDriftDataset
                data_generator = FriedmanDriftDataset(n_samples=100,
                    seed=42, change_point1=50, change_point2=75, constant=False)
                data = [data for data in data_generator]
                indices = [i for _, _, i in data]
                values = {f"x{i}": [] for i in range(5)}
                values["y"] = []
                for x, y, _ in data:
                    for i in range(5):
                        values[f"x{i}"].append(x[i])
                    values["y"].append(y)
                plt.figure(figsize=(10, 6))
                for label, series in values.items():
                    plt.plot(indices, series, label=label)
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.title('')
                plt.axvline(x=50, color='k', linestyle='--', label='Drift Point 1')
                plt.axvline(x=75, color='r', linestyle='--', label='Drift Point 2')
                plt.legend()
                plt.grid(True)
                plt.show()
        """
        self.n_samples = n_samples
        self._change_point1 = change_point1
        self._change_point2 = change_point2
        self.seed = seed
        self.index = 0
        self.rng = random.Random(self.seed)
        self.constant = constant

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.n_samples:  # Specifying end of generation
            raise StopIteration
        if self.constant:
            # x[0] is set to 1, all others to 0
            x = {0: 1}
            x.update({i: 0 for i in range(1, 10)})  # All x[i] are 0 for i > 0
        else:
            x = {i: self.rng.uniform(a=0, b=1) for i in range(10)}
        y = self._global_recurring_abrupt_gen(x, self.index) + self.rng.gauss(mu=0, sigma=1)
        result = (x, y, self.index)
        self.index += 1
        return result

    def _global_recurring_abrupt_gen(self, x, index):
        if index < self._change_point1 or index >= self._change_point2:
            return 10 * math.sin(math.pi * x[0] * x[1]) + 20 * (x[2] - 0.5) ** 2 + 10 * x[3] + 5 * x[4]
        else:
            return 10 * math.sin(math.pi * x[3] * x[5]) + 20 * (x[1] - 0.5) ** 2 + 10 * x[0] + 5 * x[2]

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.


        """
        return self.n_samples
