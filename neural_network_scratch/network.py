import numpy as np
from structlog import get_logger
from random import shuffle
from sklearn.metrics import accuracy_score
import copy

from neural_network_scratch.activation import (
    sigmoid,
    sigmoid_deriv,
    output_error,
    label_one_hot_encoded,
)
from neural_network_scratch.sample import Sample


class Network:
    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: list[int],
        number_classes: int,
    ):
        self._logger = get_logger()
        self._number_classes = number_classes

        self._layer_sizes = [input_size] + hidden_layer_sizes + [number_classes]

        self._weights = []
        bias_scaling_factor = [-36] + [1] * (len(self._layer_sizes) - 1)
        self._biases = []
        for i in range(len(self._layer_sizes) - 1):
            self._weights += [np.random.rand(self._layer_sizes[i + 1], self._layer_sizes[i])]
            self._biases += [np.random.rand(self._layer_sizes[i + 1]) * bias_scaling_factor[i]]

    def predict(self, x: np.array) -> int:
        _, activations = self._forward_propagation(x)
        return np.argmax(activations[-1])

    def _forward_propagation(self, x: np.array) -> tuple[list[np.array], list[np.array]]:
        zs = []
        activations = []

        input = x
        for weights, biases in zip(self._weights, self._biases):
            z = np.matmul(weights, input) + biases
            activation = sigmoid(z)

            zs += [copy.deepcopy(z)]
            activations += [copy.deepcopy(activation)]

            input = activation

        return zs, activations

    def train(
        self,
        samples: list[Sample],
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> list[float]:
        accuracies = []
        for epoch in range(epochs):
            accuracy = self._train_epoch(samples, batch_size, learning_rate)
            accuracies += [accuracy]

            if (epochs > 10 and epoch % int(epochs / 10) == 0) or epochs <= 10:
                self._logger.debug(f"finished epoch {epoch} with accuracy of {accuracy}")

        return accuracies

    def _train_epoch(
        self,
        samples: list[Sample],
        batch_size: int,
        learning_rate: float,
    ) -> float:
        shuffle(samples)

        test_size = 0.1
        limit = int(len(samples) * test_size)
        test_samples = samples[:limit]
        train_samples = samples[limit:]

        batches = [
            train_samples[x : x + batch_size] for x in range(0, len(train_samples), batch_size)
        ]

        for batch in batches:
            self._update_with_batch(batch, learning_rate)

        predictions = []
        labels = []
        for sample in test_samples:
            predicted = self.predict(sample.image.flatten())
            labels += [sample.label]
            predictions += [predicted]

        accuracy = accuracy_score(labels, predictions)
        return accuracy

    def _update_with_batch(self, batch: list[Sample], learning_rate: float) -> None:
        nabla_b_total = [np.zeros(b.shape) for b in self._biases]
        nabla_w_total = [np.zeros(w.shape) for w in self._weights]

        for sample in batch:
            x = sample.image.flatten()

            zs, activations = self._forward_propagation(x)
            activations = [x] + activations

            last_position = -1

            delta = output_error(
                activations[last_position],
                label_one_hot_encoded(sample.label, self._number_classes),
                zs[last_position],
            )
            nabla_b_total[last_position] += delta
            nabla_w_total[last_position] += np.outer(delta, activations[last_position - 1])

            for i in range(len(self._layer_sizes) - 2):
                delta = np.matmul(self._weights[last_position - i].T, delta) * sigmoid_deriv(
                    zs[last_position - 1 - i]
                )

                nabla_b_total[last_position - 1 - i] += delta
                nabla_w_total[last_position - 1 - i] += np.outer(
                    delta, activations[last_position - 2 - i]
                )

        for i in range(len(self._weights)):
            self._weights[i] -= learning_rate * (nabla_w_total[i] / len(batch))
            self._biases[i] -= learning_rate * (nabla_b_total[i] / len(batch))
