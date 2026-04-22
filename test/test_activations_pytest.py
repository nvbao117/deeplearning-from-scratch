import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from scipy.special import erf
from activation_functions.activation_functions import (
    sigmoid, d_sigmoid,
    tanh, d_tanh,
    relu, d_relu,
    leaky_relu, d_leaky_relu,
    gelu, d_gelu,
    swish, d_swish,
    softmax,
)


def numerical_gradient(f, x, h=1e-5):
    """Calculate numerical gradient using central difference."""
    return (f(x + h) - f(x - h)) / (2 * h)


def check_derivative(f, df, x_range=(-5, 5), n=200, h=1e-5, atol=1e-4, **kwargs):
    """Check if df matches the numerical gradient of f."""
    xs = np.linspace(x_range[0], x_range[1], n)
    numerical = np.array([numerical_gradient(f, x, h) for x in xs])
    analytical = np.array([df(x, **kwargs) for x in xs])
    return np.allclose(numerical, analytical, atol=atol)


def _exact_gelu(x):
    """Exact GELU (d_gelu is the derivative of this, not the tanh approximation)."""
    return 0.5 * x * (1 + erf(x / np.sqrt(2.0)))


# ============================================================================
# SIGMOID TESTS
# ============================================================================

class TestSigmoid:
    """Test sigmoid activation and its derivative."""

    def test_known_values(self):
        assert np.isclose(sigmoid(0.0), 0.5)
        assert np.isclose(sigmoid(np.inf), 1.0)
        assert np.isclose(sigmoid(-np.inf), 0.0, atol=1e-5)

    def test_output_range(self):
        x = np.linspace(-10, 10, 100)
        y = sigmoid(x)
        assert np.all(y > 0) and np.all(y < 1)

    def test_symmetry(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(sigmoid(x) + sigmoid(-x), 1.0)

    def test_clipping_prevents_overflow(self):
        y = sigmoid(np.array([-1000.0, 1000.0]))
        assert np.all(np.isfinite(y))

    def test_derivative_accuracy(self):
        assert check_derivative(sigmoid, d_sigmoid)

    def test_derivative_at_zero(self):
        # At x=0, d_sigmoid = 0.25
        assert np.isclose(d_sigmoid(0.0), 0.25)


# ============================================================================
# TANH TESTS
# ============================================================================

class TestTanh:
    """Test tanh activation and its derivative."""

    def test_known_values(self):
        assert np.isclose(tanh(0.0), 0.0)
        assert np.isclose(tanh(np.inf), 1.0)
        assert np.isclose(tanh(-np.inf), -1.0)

    def test_output_range(self):
        x = np.linspace(-10, 10, 100)
        y = tanh(x)
        assert np.all((y > -1) & (y < 1))

    def test_odd_function(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(tanh(-x), -tanh(x))

    def test_derivative_accuracy(self):
        assert check_derivative(tanh, d_tanh)

    def test_derivative_at_zero(self):
        assert np.isclose(d_tanh(0.0), 1.0)


# ============================================================================
# RELU TESTS
# ============================================================================

class TestReLU:
    """Test ReLU activation and its derivative."""

    def test_known_values(self):
        result = relu(np.array([-3.0, 0.0, 3.0]))
        np.testing.assert_array_equal(result, [0.0, 0.0, 3.0])

    def test_non_negative_output(self):
        x = np.linspace(-10, 10, 100)
        assert np.all(relu(x) >= 0)

    def test_identity_for_positive(self):
        x = np.array([1.0, 2.0, 5.0])
        np.testing.assert_array_equal(relu(x), x)

    @pytest.mark.parametrize("x_range", [(0.1, 5), (-5, -0.1)])
    def test_derivative_accuracy(self, x_range):
        # Skip x=0 (not differentiable there)
        assert check_derivative(relu, d_relu, x_range=x_range)

    def test_derivative_values(self):
        result = d_relu(np.array([-1.0, 1.0]))
        np.testing.assert_array_equal(result, [0.0, 1.0])


# ============================================================================
# LEAKY RELU TESTS
# ============================================================================

class TestLeakyReLU:
    """Test Leaky ReLU activation and its derivative."""

    def test_known_values(self):
        x = np.array([-2.0, 0.0, 2.0])
        np.testing.assert_allclose(leaky_relu(x), [-0.02, 0.0, 2.0])

    def test_custom_alpha(self):
        x = np.array([-2.0, 2.0])
        np.testing.assert_allclose(leaky_relu(x, alpha=0.1), [-0.2, 2.0])

    def test_no_dead_neurons(self):
        x = np.linspace(-10, -0.01, 50)
        assert np.all(leaky_relu(x) < 0)

    @pytest.mark.parametrize("alpha", [0.01, 0.1, 0.3])
    def test_derivative_with_different_alphas(self, alpha):
        assert check_derivative(
            lambda x: leaky_relu(x, alpha=alpha),
            lambda x: d_leaky_relu(x, alpha=alpha),
            x_range=(0.1, 5),
        )

    def test_derivative_values(self):
        result = d_leaky_relu(np.array([-1.0, 1.0]))
        np.testing.assert_allclose(result, [0.01, 1.0])


# ============================================================================
# GELU TESTS
# ============================================================================

class TestGELU:
    """Test GELU activation and its derivative."""

    def test_output_at_zero(self):
        assert np.isclose(gelu(0.0), 0.0)

    def test_monotonic_increasing(self):
        x = np.linspace(1, 10, 50)
        y = gelu(x)
        assert np.all(np.diff(y) > 0)

    def test_output_negative_for_negative_input(self):
        assert gelu(-0.5) < 0.0

    def test_tanh_approximation_close_to_exact(self):
        x = np.linspace(-5, 5, 200)
        np.testing.assert_allclose(gelu(x), _exact_gelu(x), atol=1e-3)

    def test_derivative_accuracy(self):
        # d_gelu is the derivative of exact GELU (erf-based), not tanh approx
        assert check_derivative(_exact_gelu, d_gelu, atol=1e-4)

    def test_shape_preserved(self):
        x = np.random.randn(3, 4)
        assert gelu(x).shape == x.shape
        assert d_gelu(x).shape == x.shape


# ============================================================================
# SWISH TESTS
# ============================================================================

class TestSwish:
    """Test Swish activation and its derivative."""

    def test_output_at_zero(self):
        assert np.isclose(swish(0.0), 0.0)

    def test_approximates_relu_for_large_positive(self):
        x = np.array([10.0])
        np.testing.assert_allclose(swish(x), relu(x), atol=1e-3)

    def test_slight_negative_for_negative_input(self):
        assert swish(-0.5) < 0.0

    def test_derivative_accuracy(self):
        assert check_derivative(swish, d_swish, atol=1e-4)

    def test_shape_preserved(self):
        x = np.random.randn(3, 4)
        assert swish(x).shape == x.shape
        assert d_swish(x).shape == x.shape


# ============================================================================
# SOFTMAX TESTS
# ============================================================================

class TestSoftmax:
    """Test softmax activation."""

    def test_output_sums_to_one(self):
        x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        y = softmax(x)
        np.testing.assert_allclose(y.sum(axis=1), [1.0, 1.0])

    def test_uniform_input_gives_uniform_output(self):
        x = np.ones((2, 4))
        y = softmax(x)
        np.testing.assert_allclose(y, np.full((2, 4), 0.25))

    def test_output_in_probability_range(self):
        x = np.random.randn(5, 10)
        y = softmax(x)
        assert np.all(y > 0) and np.all(y < 1)

    def test_numerical_stability_large_values(self):
        x = np.array([[1000.0, 1001.0, 1002.0]])
        y = softmax(x)
        assert np.all(np.isfinite(y))
        np.testing.assert_allclose(y.sum(axis=1), [1.0])

    def test_argmax_preserved(self):
        x = np.array([[0.1, 5.0, 0.3]])
        y = softmax(x)
        assert np.argmax(y) == 1

    def test_output_shape(self):
        x = np.random.randn(4, 6)
        assert softmax(x).shape == (4, 6)


# ============================================================================
# PARAMETRIZED TESTS (chạy hàm trên nhiều input khác nhau)
# ============================================================================

@pytest.mark.parametrize("x,expected", [
    (0.0, 0.5),
    (-1000, 0.0),
    (1000, 1.0),
])
def test_sigmoid_edge_cases(x, expected):
    """Test sigmoid with various edge cases."""
    assert np.isclose(sigmoid(x), expected, atol=1e-5)


@pytest.mark.parametrize("func,d_func", [
    (sigmoid, d_sigmoid),
    (tanh, d_tanh),
    (swish, d_swish),
])
def test_derivative_at_zero(func, d_func):
    """Test that functions are continuous at zero."""
    x = 1e-6
    fd_approx = (func(x) - func(-x)) / (2 * x)
    assert np.isclose(d_func(0.0), fd_approx, atol=1e-4)
