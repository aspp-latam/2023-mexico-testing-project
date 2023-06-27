# from logistic_map import logistic_map, iterate_f
from logistic_map import *
from math import isclose
from numpy.testing import assert_array_almost_equal
import numpy as np
import pytest

SEED = np.random.randint(0, 2**31)

@pytest.fixture
def random_state():
    print(f'Using random seed: {SEED}')
    rs = np.random.RandomState(SEED)
    return rs

def test_something(random_state):
    random_state.rand()

@pytest.mark.parametrize("x, r, expected", 
                         [
                          (0.1, 2.2, 0.198), 
                          (0.2, 3.4, 0.544), 
                          (0.75, 1.7, 0.31875)
                          ]
                         )
def test_logistic_map(x, r, expected):
    assert isclose(logistic_map(x, r), expected)

@pytest.mark.parametrize("x, r, it, expected", 
                         [(0.1, 2.2, 1, [0.198]), 
                          (0.2, 3.4, 4, [0.544, 0.843418, 0.449019, 0.841163]), 
                          (0.75, 1.7, 2, [0.31875, 0.369152])
                          ]
                         )
def test_iterations(x, r, it, expected):
    iterations = iterate_f(it, x, r)
    print(iterations)
    assert_array_almost_equal(iterations, expected, decimal=6)

#SEED = 42
#random_state = np.random.RandomState(SEED)

#n_vals = 10
#x0 = random_state.rand(n_vals)
#
#@pytest.mark.parametrize("x0", x0)
#def test_random_x0(x0, r=1.5, it=100, expected=1/3):
#    iterations = iterate_f(it, x0, r)
#    last_x = iterations[-1]
#    assert isclose(last_x, expected, abs_tol=0.01)
#

def test_converges(random_state, r=1.5, it=100, expected=1/3):
    n_vals = 10
    x0 = random_state.rand()
    iterations = iterate_f(it, x0, r)
    last_x = iterations[-1]
    assert isclose(last_x, expected, abs_tol=0.01)
