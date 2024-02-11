import numpy as np

from neural_network_scratch.activation import label_one_hot_encoded


def test_label_one_hot_encoded() -> None:
    y = 4
    num_classes = 10
    out = label_one_hot_encoded(y, num_classes)
    assert (out == np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])).all()
