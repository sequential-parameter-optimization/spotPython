class ListGenerator:
    def __init__(self, hparams, L_in, L_out):
        self.hparams = hparams
        self._L_in = L_in
        self._L_out = L_out

    def _get_hidden_sizes(self) -> list:
        """
        Generate the hidden layer sizes for the network based on nn_shape.

        Returns:
            list: A list of hidden layer sizes.
        """
        n_low = self._L_in // 4  # Minimum number of neurons
        # n_high = max(self.hparams.l1, 2 * n_low)  # Maximum number of neurons

        # TODO: Überlegen, wie rum es besser ist
        if self.hparams.l_n > self.hparams.l1:
            self.hparams.l1 = self.hparams.l_n
            # raise ValueError("l_n must be bigger than l1")

        if self.hparams.nn_shape == "Funnel":
            step_size = (self.hparams.l1 - self._L_out) // self.hparams.l_n
            hidden_sizes = list(range(self.hparams.l1, self._L_out, -step_size))

        elif self.hparams.nn_shape == "Diamond":
            mid_point = (self.hparams.l_n + 1) // 2
            increasing_part = [self.hparams.l1]
            for _ in range(1, mid_point):
                next_size = int(increasing_part[-1] * 1.2)
                increasing_part.append(next_size)

            remaining_layers = self.hparams.l_n - mid_point
            step_size = (increasing_part[-1] - self._L_out) // (remaining_layers + 1)

            decreasing_part = []
            current_size = increasing_part[-1]
            for _ in range(remaining_layers):
                current_size = max(self._L_out, current_size - step_size)
                decreasing_part.append(current_size)

            hidden_sizes = increasing_part + decreasing_part

        elif self.hparams.nn_shape == "Hourglass":
            mid_point = (self.hparams.l_n) // 2
            step_size = (self.hparams.l1 - n_low) // (mid_point - 1)

            decreasing_part = [self.hparams.l1]
            for _ in range(1, mid_point):
                next_size = decreasing_part[-1] - step_size
                decreasing_part.append(max(n_low, next_size))

            increasing_part = [decreasing_part[-1] + step_size]
            for _ in range(mid_point, self.hparams.l_n - 2):
                next_size = increasing_part[-1] + step_size
                increasing_part.append(min(self.hparams.l1, next_size))

            last_step_size = (increasing_part[-1] - self._L_out) // 2
            decreasing_to_output = max(self._L_out, increasing_part[-1] - last_step_size)

            hidden_sizes = decreasing_part + increasing_part + [decreasing_to_output]

        elif self.hparams.nn_shape == "Wave":
            half_wave = (self.hparams.l_n) // 4
            step_size = (self.hparams.l1 - n_low) // (half_wave - 1)

            decreasing_part_1 = [self.hparams.l1]
            for _ in range(1, half_wave):
                next_size = decreasing_part_1[-1] - step_size
                decreasing_part_1.append(max(n_low, next_size))

            increasing_part_1 = [decreasing_part_1[-1] + step_size]
            for _ in range(half_wave, 2 * half_wave - 1):
                next_size = increasing_part_1[-1] + step_size
                increasing_part_1.append(next_size)

            decreasing_part_2 = [increasing_part_1[-1] - step_size]
            for _ in range(2 * half_wave, 3 * half_wave - 1):
                next_size = decreasing_part_2[-1] - step_size
                decreasing_part_2.append(max(n_low, next_size))

            increasing_part_2 = [decreasing_part_2[-1] + step_size]
            for _ in range(3 * half_wave, self.hparams.l_n - 2):
                next_size = increasing_part_2[-1] + step_size
                increasing_part_2.append(next_size)

            last_step_size = (increasing_part_2[-1] - self._L_out) // 2
            decreasing_to_output = max(self._L_out, increasing_part_2[-1] - last_step_size)

            hidden_sizes = decreasing_part_1 + increasing_part_1 + decreasing_part_2 + increasing_part_2 + [decreasing_to_output]

        elif self.hparams.nn_shape == "Block":
            hidden_sizes = [self.hparams.l1] * self.hparams.l_n

        else:
            raise ValueError(f"Unknown nn_shape: {self.hparams.nn_shape}")

        return hidden_sizes
