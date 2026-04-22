import numpy as np


# ─────────────────────────── helpers ────────────────────────────────────────

def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def cosine_similarity(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom < 1e-10 else float(np.dot(a, b) / denom)


# ═════════════════════════ REGRESSION LOSSES ════════════════════════════════

def mse(predictions, targets):
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return float(np.mean((predictions - targets) ** 2))


def mse_gradient(predictions, targets):
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return 2.0 * (predictions - targets) / predictions.size


def mae(predictions, targets):
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return float(np.mean(np.abs(predictions - targets)))


def mae_gradient(predictions, targets):
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return np.sign(predictions - targets) / predictions.size


def huber(predictions, targets, delta=1.0):
    """Smooth L1: quadratic for |error| < delta, linear otherwise."""
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    err = np.abs(predictions - targets)
    loss = np.where(err <= delta, 0.5 * err ** 2, delta * (err - 0.5 * delta))
    return float(np.mean(loss))


def huber_gradient(predictions, targets, delta=1.0):
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    diff = predictions - targets
    grad = np.where(np.abs(diff) <= delta, diff, delta * np.sign(diff))
    return grad / predictions.size


def log_cosh(predictions, targets):
    """Differentiable everywhere; behaves like MSE for small errors, MAE for large."""
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return float(np.mean(np.log(np.cosh(predictions - targets))))


def log_cosh_gradient(predictions, targets):
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return np.tanh(predictions - targets) / predictions.size


# ═════════════════════════ CLASSIFICATION LOSSES ════════════════════════════

def binary_cross_entropy(predictions, targets, eps=1e-15):
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    p = np.clip(predictions, eps, 1 - eps)
    return float(-np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p)))


def bce_gradient(predictions, targets, eps=1e-15):
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    p = np.clip(predictions, eps, 1 - eps)
    return (-(targets / p) + (1 - targets) / (1 - p)) / predictions.size


def categorical_cross_entropy(logits, target_index, eps=1e-15):
    probs = softmax(np.asarray(logits, dtype=float))
    return float(-np.log(np.clip(probs[target_index], eps, 1.0)))


def cce_gradient(logits, target_index):
    probs = softmax(np.asarray(logits, dtype=float))
    grad = probs.copy()
    grad[target_index] -= 1.0
    return grad


def label_smoothed_cce(logits, target_index, num_classes, alpha=0.1, eps=1e-15):
    probs = softmax(np.asarray(logits, dtype=float))
    smooth = np.full(num_classes, alpha / num_classes)
    smooth[target_index] = 1.0 - alpha + alpha / num_classes
    return float(-np.sum(smooth * np.log(np.clip(probs, eps, 1.0))))


def label_smoothed_cce_gradient(logits, target_index, num_classes, alpha=0.1):
    probs = softmax(np.asarray(logits, dtype=float))
    smooth = np.full(num_classes, alpha / num_classes)
    smooth[target_index] = 1.0 - alpha + alpha / num_classes
    return probs - smooth


def focal_loss(predictions, targets, gamma=2.0, eps=1e-15):
    """Down-weights easy examples to focus training on hard ones."""
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    p = np.clip(predictions, eps, 1 - eps)
    pt = np.where(targets == 1, p, 1 - p)
    return float(-np.mean((1 - pt) ** gamma * np.log(pt)))


def focal_loss_gradient(predictions, targets, gamma=2.0, eps=1e-15):
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    p = np.clip(predictions, eps, 1 - eps)
    pt = np.where(targets == 1, p, 1 - p)
    log_pt = np.log(pt)
    sign = np.where(targets == 1, 1.0, -1.0)
    grad = sign * (-(1 - pt) ** gamma * (1 / pt) - gamma * (1 - pt) ** (gamma - 1) * log_pt * (-1))
    return grad / predictions.size


def hinge(predictions, targets):
    """Binary SVM loss. targets must be in {-1, +1}."""
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return float(np.mean(np.maximum(0.0, 1.0 - targets * predictions)))


def hinge_gradient(predictions, targets):
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    mask = (targets * predictions < 1).astype(float)
    return -targets * mask / predictions.size


def multiclass_hinge(scores, target_index, margin=1.0):
    """Crammer-Singer multi-class SVM loss."""
    scores = np.asarray(scores, dtype=float)
    correct = scores[target_index]
    margins = np.maximum(0.0, scores - correct + margin)
    margins[target_index] = 0.0
    return float(np.sum(margins))


def multiclass_hinge_gradient(scores, target_index, margin=1.0):
    scores = np.asarray(scores, dtype=float)
    correct = scores[target_index]
    mask = ((scores - correct + margin) > 0).astype(float)
    mask[target_index] = 0.0
    grad = mask.copy()
    grad[target_index] = -np.sum(mask)
    return grad


# ═════════════════════════ DIVERGENCE / DENSITY ═════════════════════════════

def kl_divergence(p, q, eps=1e-15):
    """KL(P || Q) — P is true distribution, Q is approximation."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def kl_divergence_gradient(p, q, eps=1e-15):
    """Gradient w.r.t. Q."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    q = np.clip(q, eps, 1.0)
    p = np.clip(p, eps, 1.0)
    return -p / q


def dice_loss(predictions, targets, smooth=1.0):
    """Segmentation loss; predictions and targets are flat probability arrays."""
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    intersection = np.sum(predictions * targets)
    return 1.0 - (2.0 * intersection + smooth) / (
        np.sum(predictions) + np.sum(targets) + smooth
    )


def dice_loss_gradient(predictions, targets, smooth=1.0):
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    numerator = 2.0 * np.sum(predictions * targets) + smooth
    denominator = np.sum(predictions) + np.sum(targets) + smooth
    return (2.0 * targets * denominator - 2.0 * numerator) / (denominator ** 2)


# ═════════════════════════ EMBEDDING / METRIC LOSSES ════════════════════════

def contrastive_loss(anchor, positive, negatives, temperature=0.07):
    """InfoNCE-style contrastive loss."""
    anchor = np.asarray(anchor, dtype=float)
    positive = np.asarray(positive, dtype=float)
    negatives = [np.asarray(n, dtype=float) for n in negatives]
    sim_pos = cosine_similarity(anchor, positive) / temperature
    sim_negs = [cosine_similarity(anchor, n) / temperature for n in negatives]
    max_sim = max(sim_pos, max(sim_negs, default=sim_pos))
    exp_pos = np.exp(sim_pos - max_sim)
    exp_negs = sum(np.exp(s - max_sim) for s in sim_negs)
    return float(-np.log(max(1e-15, exp_pos / (exp_pos + exp_negs))))


def triplet_loss(anchor, positive, negative, margin=1.0):
    """Max-margin triplet: push anchor closer to positive than negative."""
    anchor = np.asarray(anchor, dtype=float)
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    d_pos = float(np.linalg.norm(anchor - positive))
    d_neg = float(np.linalg.norm(anchor - negative))
    return float(max(0.0, d_pos - d_neg + margin))


def triplet_loss_gradients(anchor, positive, negative, margin=1.0):
    """Returns (grad_anchor, grad_positive, grad_negative)."""
    anchor = np.asarray(anchor, dtype=float)
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    d_pos = float(np.linalg.norm(anchor - positive))
    d_neg = float(np.linalg.norm(anchor - negative))
    if d_pos - d_neg + margin <= 0:
        zeros = np.zeros_like(anchor)
        return zeros, zeros, zeros
    eps = 1e-10
    g_ap = (anchor - positive) / max(d_pos, eps)
    g_an = (anchor - negative) / max(d_neg, eps)
    grad_anchor = g_ap - g_an
    grad_positive = -g_ap
    grad_negative = g_an
    return grad_anchor, grad_positive, grad_negative


# ═════════════════════════ REGULARIZATION ═══════════════════════════════════

def l1_regularization(weights, lam=1e-4):
    return float(lam * np.sum(np.abs(weights)))


def l1_gradient(weights, lam=1e-4):
    return lam * np.sign(np.asarray(weights, dtype=float))


def l2_regularization(weights, lam=1e-4):
    return float(lam * np.sum(np.asarray(weights, dtype=float) ** 2))


def l2_gradient(weights, lam=1e-4):
    return 2.0 * lam * np.asarray(weights, dtype=float)


def elastic_net(weights, lam=1e-4, rho=0.5):
    """Convex combination of L1 and L2: rho * L1 + (1-rho) * L2."""
    w = np.asarray(weights, dtype=float)
    return float(lam * (rho * np.sum(np.abs(w)) + (1 - rho) * np.sum(w ** 2)))


def elastic_net_gradient(weights, lam=1e-4, rho=0.5):
    w = np.asarray(weights, dtype=float)
    return lam * (rho * np.sign(w) + 2 * (1 - rho) * w)
