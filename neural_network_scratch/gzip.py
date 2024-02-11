from array import array
from struct import unpack
import gzip
import numpy as np


def parse_gzip(f):
    data_types = {
        0x08: "B",
        0x09: "b",
        0x0B: "h",
        0x0C: "i",
        0x0D: "f",
        0x0E: "d",
    }

    header = f.read(4)
    _, data_type_label, num_dimensions = unpack(">HBB", header)
    data_type_label = data_types[data_type_label]
    dimension_sizes = unpack(">" + "I" * num_dimensions, f.read(4 * num_dimensions))
    data = array(data_type_label, f.read())
    data.byteswap()

    return np.array(data).reshape(dimension_sizes)


def read_gzip_from_path(path: str) -> np.array:
    with gzip.open(path, "rb") as f:
        return parse_gzip(f)
