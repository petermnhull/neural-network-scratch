from dataclasses import dataclass
import numpy as np


@dataclass
class Sample:
    image: np.array
    label: int
