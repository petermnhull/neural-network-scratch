from neural_network_scratch import Network, Sample

import numpy as np
from random import randint


def test_train():
    num_classes = 2
    input_layer = 100

    network = Network(
        input_size=input_layer,
        hidden_layer_sizes=[16, 16],
        number_classes=num_classes,
    )

    inputs = [np.random.rand(input_layer) for _ in range(20)]
    labels = [randint(0, 1) for _ in range(20)]
    train = [Sample(image=x, label=label) for (x, label) in zip(inputs, labels)]

    epochs = 3
    accuracies = network.train(
        samples=train,
        epochs=epochs,
        batch_size=1,
        learning_rate=0.5,
    )

    # Check there's accuracies for each epoch
    assert len(accuracies) == epochs
    assert 0 <= accuracies[0] <= 1

    # Check prediction for random input
    x4 = np.random.rand(input_layer)
    predicted = network.predict(x4)
    assert 0 <= predicted <= 1
