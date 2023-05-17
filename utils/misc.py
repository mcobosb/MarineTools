import os

from marinetools.utils import read


def precipitation():
    path_ = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "misc",
            "precipitation.zip",
        )
    return read.csv(path_, ts=True)
