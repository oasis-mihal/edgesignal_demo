import numpy as np


def softmax(results: np.ndarray) -> np.ndarray:
    """
    Softmaxes a result set
    Args:
        results: Array of results to softmax

    Returns:
        Softmaxed array
    """
    return np.exp(results) / np.sum(np.exp(results))