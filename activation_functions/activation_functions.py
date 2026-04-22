import numpy as np
from scipy.special import erf


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    t = tanh(x)
    return 1 - t * t


def relu(x):
    return np.maximum(x, 0.0)


def d_relu(x):
    return (x > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def d_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def d_gelu(x):
    phi = 0.5 * (1 + erf(x / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * np.square(x)) / np.sqrt(2.0 * np.pi)
    return phi + x * pdf


def swish(x):
    return x * sigmoid(x)


def d_swish(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)


def softmax(xs):
    max_x = np.max(xs, axis=1, keepdims=True)
    exp_xs = np.exp(xs - max_x)
    return exp_xs / np.sum(exp_xs, axis=1, keepdims=True)
