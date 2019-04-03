# Copyright 2019 the wasserstein_fairness Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Basic components of the costs that combine in Wasserstein Fairness.

The functions in this file are centred around the calculation of individual
costs/losses. Functions that combine these costs (and gradients thereof) appear
in `combined_costs.py`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sparse
import scipy.special as scisp
import sklearn.metrics as skmet


# Use scipy.special's expit for the logistic; it's numerically stable.
sigmoid = scisp.expit


def regression_loss(y_pred, y_true):
  """Logistic regression loss."""
  # Aldo's logistic regression loss is the same loss that scikit-learn expresses
  # but with the arguments reversed (and some numerical precautions absent).
  return skmet.log_loss(y_true, y_pred)


def regression_loss_gradient(x, y, theta):
  """Gradient of the logistic loss with respect to theta.

  The regression parameters `theta` may have one more entry than there are
  features (columns) in `x`, in which case an intercept term will be added to
  `x` prior to computing the gradient.

  Args:
    x: Feature inputs to the regression (each row a data point). An MxN matrix.
    y: Target outputs of the regression; a vector of length M.
    theta: Regression parameters; a vector of length N or N+1.

  Returns:
    The gradient as described.
  """
  x = _maybe_add_intercept(x, theta)
  h = sigmoid(np.dot(x, theta))         # was matmul
  return np.dot(x.T, (h - y)) / y.size  # was matmul


def wasserstein_two_loss_gradient(x1, x2, coupling, theta):
  """Gradient of the Wasserstein-2 distance loss with respect to theta.

  The regression parameters `theta` may have one more entry than there are
  features (columns) in `x1` or `x2`, in which case an intercept term will be
  added to both prior to computing the gradient.

  Args:
    x1: A subset of feature inputs to the regression: a JxN matrix.
    x2: Another subset of feature inputs to the regression: a KxN matrix.
    coupling: A JxK Wasserstein distance coupling matrix of the kind computed by
        `optimal_transport.wasserstein_coupling_on_the_real_line`. Can be a
        scipy.sparse sparse matrix.
    theta: Regression parameters; a vector of length N or N+1.

  Returns:
    The gradient as described.
  """
  x1 = _maybe_add_intercept(x1, theta)
  x2 = _maybe_add_intercept(x2, theta)

  weights_1 = np.asarray(np.sum(coupling, axis=1)).ravel()
  weights_2 = np.asarray(np.sum(coupling, axis=0)).ravel()
  # The prior two lines used to be as follows, but are revised to the above to
  # allow use of sparse matrices.
  # weights_1 = np.sum(coupling, axis=1)
  # weights_2 = np.sum(coupling, axis=0)

  h1 = sigmoid(np.dot(x1, theta))  # was matmul
  h2 = sigmoid(np.dot(x2, theta))  # was matmul

  nh1p1 = 1 - h1  # The name means "negative h1 plus 1".
  nh2p1 = 1 - h2

  term_1 = 2 * np.dot(weights_1 * nh1p1 * h1**2, x1)  # was matmul
  term_2 = 2 * np.dot(weights_2 * nh2p1 * h2**2, x2)  # was matmul
  term_3 = -2 * (np.dot(coupling.dot(h2) * h1 * nh1p1, x1) +  # was matmul
                 np.dot(coupling.T.dot(h1) * h2 * nh2p1, x2))   # was matmul
        # The last line of term_3 used to be as follows, but is revised to the
        # above to allow use of sparse matrices.
        #   np.dot(np.dot(h1, coupling) * h2 * nh2p1, x2))   # was matmul

  return term_1 + term_2 + term_3


def wass_one_barycenter_loss_gradient(b, x2, coupling, theta, delta):
  """Gradient of the Wass1 barycenter loss with respect to theta.

  Args:
    b: barycenter distribution as a column Lx1 of scores.
    x2: A subset of feature inputs to the regression: a KxN matrix.
    coupling: A LxK Wasserstein distance coupling matrix of the kind computed by
        `optimal_transport.wasserstein_coupling_on_the_real_line`. Can be a
        scipy.sparse sparse matrix.
    theta: Regression parameters; a vector of length N or N+1.
    delta: pseudo-Huber loss parameter.

  Returns:
    The gradient as described.
  """
  x2 = _maybe_add_intercept(x2, theta)
  h2 = sigmoid(np.dot(x2, theta))  # was matmul
  mask = (coupling != 0.0)
  if sparse.issparse(coupling):
    diff = (mask.dot(sparse.spdiags(h2, 0, len(h2), len(h2))) -
            sparse.spdiags(b, 0, len(b), len(b)).dot(mask))
  else:
    diff = h2.reshape((1, -1)) - b.reshape((-1, 1))

  denom = (diff / delta).power(2.0)    # For dense matrices, entries in denom
  denom[mask] += 1.0                   # and d_huber_d_diff that correspond to
  denom = np.sqrt(denom)               # zero entries in mask will be nonsense
                                       # and should be ignored. The elementwise
  d_huber_d_diff = diff.copy()         # multiplication with coupling below will
  d_huber_d_diff[mask] /= denom[mask]  # zero out those entries.

  if sparse.issparse(coupling):
    coupling_d_huber_d_diff = coupling.multiply(d_huber_d_diff)
  else:
    coupling_d_huber_d_diff = np.multiply(coupling, d_huber_d_diff)

  d_h2_d_logits2 = h2 * (1 - h2)

  # Now we create weight vectors that measure how much the sigmoid and pseudo-
  # Huber derivatives weight the data points (which are the derivatives of the
  # logits with respect to theta.
  weights2 = (d_h2_d_logits2 *
              np.asarray(np.sum(coupling_d_huber_d_diff, axis=0)).ravel())

  # Finally, the gradient of theta itself.
  return np.dot(weights2, x2)


def wasserstein_one_loss_gradient_method_one(x1, x2, coupling, theta, delta):
  """Gradient of the Wasserstein-1 distance loss with respect to theta.

  To ensure differentiability, we substitute a pseudo-Huber loss term for the
  absolute value difference used by the true Wasserstein-1 gradient. The delta
  parameter controls how curvy the minimum is---but also controls the how steep
  the gradient is far away from 0. See Wikipedia for more details:
  https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function.

  The regression parameters `theta` may have one more entry than there are
  features (columns) in `x1` or `x2`, in which case an intercept term will be
  added to both prior to computing the gradient.

  The `wasserstein_one_loss_gradient_method_two` function is an alternative
  implementation of this method.

  Args:
    x1: A subset of feature inputs to the regression: a JxN matrix.
    x2: Another subset of feature inputs to the regression: a KxN matrix.
    coupling: A JxK Wasserstein distance coupling matrix of the kind computed by
        `optimal_transport.wasserstein_coupling_on_the_real_line`. Must be a
        scipy.sparse sparse matrix.
    theta: Regression parameters; a vector of length N or N+1.
    delta: The pseudo-Huber loss delta parameter.

  Returns:
    The gradient as described.

  Raises:
    TypeError: coupling is not a scipy.sparse matrix.
  """
  if not sparse.issparse(coupling): raise TypeError(
      'The `coupling` argument to `wasserstein_one_loss_gradient` must be '
      'a scipy.sparse sparse matrix.')

  x1 = _maybe_add_intercept(x1, theta)
  x2 = _maybe_add_intercept(x2, theta)

  # The predictions for the inputs:
  logits1 = np.dot(x1, theta)
  logits2 = np.dot(x2, theta)
  h1 = sigmoid(logits1)
  h2 = sigmoid(logits2)

  # First, we compute a matrix where element j,k expresses:
  #     coupling[j,k] * diff[j,k] / sqrt(1 + (diff[j,k] / delta)^2)
  # where
  #     diff[j,k] = h1[j] - h2[k].
  # If coupling is sparse, we only compute diff entries where coupling != 0.
  mask = (coupling != 0.0)
  if sparse.issparse(coupling):
    diff = (sparse.spdiags(h1, 0, len(h1), len(h1)).dot(mask) -
            mask.dot(sparse.spdiags(h2, 0, len(h2), len(h2))))
  else:
    diff = h1.reshape((-1, 1)) - h2.reshape((1, -1))

  denom = (diff / delta).power(2.0)    # For dense matrices, entries in denom
  denom[mask] += 1.0                   # and d_huber_d_diff that correspond to
  denom = np.sqrt(denom)               # zero entries in mask will be nonsense
                                       # and should be ignored. The elementwise
  d_huber_d_diff = diff.copy()         # multiplication with coupling below will
  d_huber_d_diff[mask] /= denom[mask]  # zero out those entries.

  if sparse.issparse(coupling):
    coupling_d_huber_d_diff = coupling.multiply(d_huber_d_diff)
  else:
    coupling_d_huber_d_diff = np.multiply(coupling, d_huber_d_diff)

  # Differentiate the sigmoids w.r.t. their arguments. This is required for
  # differentiation of diff, which is required by the chain rule.
  d_h1_d_logits1 = h1 * (1 - h1)
  d_h2_d_logits2 = h2 * (1 - h2)

  # Now we create weight vectors that measure how much the sigmoid and pseudo-
  # Huber derivatives weight the data points (which are the derivatives of the
  # logits with respect to theta.
  weights1 = (d_h1_d_logits1 *
              np.asarray(np.sum(coupling_d_huber_d_diff, axis=1)).ravel())
  weights2 = (d_h2_d_logits2 *
              np.asarray(np.sum(coupling_d_huber_d_diff, axis=0)).ravel())

  # Finally, the gradient of theta itself.
  term1 = np.dot(weights1, x1)
  term2 = np.dot(weights2, x2)

  return term1 - term2


def wasserstein_one_loss_gradient_method_two(x1, x2, coupling, theta, delta):
  """Compute Wasserstein-1 loss gradient using the Wass-2 gradient.

  To ensure differentiability, we substitute a pseudo-Huber loss term for the
  absolute value difference used by the true Wasserstein-1 gradient. The delta
  parameter controls how curvy the minimum is---but also controls the how steep
  the gradient is far away from 0. See Wikipedia for more details:
  https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function.

  The regression parameters `theta` may have one more entry than there are
  features (columns) in `x1` or `x2`, in which case an intercept term will be
  added to both prior to computing the gradient.

  The `wasserstein_one_loss_gradient_method_one` function is an alternative
  implementation of this method.

  Args:
    x1: A subset of feature inputs to the regression: a JxN matrix.
    x2: Another subset of feature inputs to the regression: a KxN matrix.
    coupling: A JxK Wasserstein distance coupling matrix of the kind computed by
        `optimal_transport.wasserstein_coupling_on_the_real_line`. Must be a
        scipy.sparse sparse matrix.
    theta: Regression parameters; a vector of length N or N+1.
    delta: The pseudo-Huber loss delta parameter.

  Returns:
    The gradient as described.

  Raises:
    TypeError: coupling is not a scipy.sparse matrix.
  """
  xx1 = _maybe_add_intercept(x1, theta)
  xx2 = _maybe_add_intercept(x2, theta)
  h1 = sigmoid(np.dot(xx1, theta))
  h2 = sigmoid(np.dot(xx2, theta))

  mask = (coupling != 0.0)
  if sparse.issparse(coupling):
    diff = (sparse.spdiags(h1, 0, len(h1), len(h1)).dot(mask) -
            mask.dot(sparse.spdiags(h2, 0, len(h2), len(h2))))
  else:
    diff = h1.reshape((-1, 1)) - h2.reshape((1, -1))

  denom = (diff / delta).power(2.0)
  denom[mask] += 1.0
  denom = np.sqrt(denom)
  multiplier = (0.5 * denom.power(-1))
  return wasserstein_two_loss_gradient(
      x1, x2, coupling.multiply(multiplier), theta)


def predict_prob(x, theta):
  """Predict probabilities for input datapoints x."""
  x = _maybe_add_intercept(x, theta)
  return sigmoid(np.dot(x, theta))  # was matmul


def predict(x, theta, threshold):
  """Predict outcomes for input datapoints x via thresholding."""
  return predict_prob(x, theta) > threshold


### Private helpers ###


def _maybe_add_intercept(x, theta):
  """Append intercept column to feature data `x` if `theta` needs one.

  Also converts x to a numpy array if needed.

  Args:
    x: Feature matrix; MxN. Each row is a data point.
    theta: Regression parameters; a vector of length N or N+1.

  Returns:
    `x` if `x` is as wide as `theta` is long; otherwise, `x` with an
    additional intercept column of ones.

  Raises:
    ValueError: `len(theta)` isn't in `[len(x[0]), len(x[0]) + 1]`.
  """
  x = np.asarray(x)
  theta = np.asarray(theta)
  if x.shape[1] == theta.shape[-1] - 1:
    return _add_intercept(x)
  elif x.shape[1] == theta.shape[-1]:
    return x
  else:
    raise ValueError(
        'Shape mismatch when deciding whether to add an intercept column to '
        'the data: x.shape={}, theta.shape={}'.format(x.shape, theta.shape))


def _add_intercept(x):
  """Append an intercept column to the feature data `x`."""
  return np.pad(x, ((0, 0), (0, 1)), mode='constant',
                constant_values=((0., 0.), (0., 1.)))
