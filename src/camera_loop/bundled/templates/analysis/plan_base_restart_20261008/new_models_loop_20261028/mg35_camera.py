"""Match Matrix-Game's float32 C2W storage followed by float64 inversion."""
import numpy as np
from common import rejoin


def native_w2c(commands):
    return np.linalg.inv(np.asarray(commands, np.float32).astype(np.float64)).astype(np.float32)


def corrected_commands(measured, previous, future):
    desired = rejoin(measured, previous, future)
    # An identity intervention must not introduce inverse/matmul roundoff.
    if np.array_equal(measured, previous):
        return desired
    return (previous @ np.linalg.inv(measured)) @ desired
