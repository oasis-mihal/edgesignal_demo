import numpy as np

from src.HandyFunctions import softmax

EPSILON = 1e-4
def test_softmax():
    assert np.isclose(softmax([0, 1.0]), [0.26894, 0.73105], rtol=EPSILON).all()
    assert np.isclose(softmax([1.0, 0.0]), [0.73106, 0.26894], rtol=EPSILON).all()
    assert np.isclose(softmax([10.0, 0.5]), [9.99925e-01, 7.48462e-05], rtol=EPSILON).all()