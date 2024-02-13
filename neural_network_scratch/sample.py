from dataclasses import dataclass
import numpy as np


@dataclass
class Sample:
    values: np.array
    label: int
