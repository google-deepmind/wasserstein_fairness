# coding=utf8

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

"""Algorithms relating to optimal transport."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl import logging
import numpy as np
import scipy.sparse as sparse


# The inner calculation of the sorted Wasserstein coupling in
# wasserstein_coupling_on_the_real_line depends only on the lengths of the
# vector arguments to that function. We can cache those couplings and
# avoid a costly loop.
_WCOTRL_CACHE = {}


def sinkhorn(cost, lambda_, distn_a, distn_b, tolerance=0.0000001):
  """Use the Sinkhorn algorithm to find an optimum coupling.

  An implementation of the algorithm described in
  https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport.pdf
  with variable names referring to notation found there.

  Args:
    cost: I am pretty sure that this matrix should be square (NxN) and strictly
        positive.
    lambda_: Regulariser.
    distn_a: Input distribution a. A vector of length N.
    distn_b: Input distribution b. A vector of length N.
    tolerance: Stopping tolerance for the iteration.

  Returns:
    A 3-tuple with the following members:
    [0]: the left matrix scaling factor u (size: [N]),
    [1]: the right matrix scaling factor v (size: [N]),
    [2]: the optimum coupling between distn_a and distn_b.
  """
  big_k = np.exp(-lambda_ * cost)
  big_k_tilde = np.dot(np.diag(1.0 / distn_a), big_k)  # was matmul

  u = np.ones_like(distn_a) / len(distn_a)
  old_u = u + tolerance

  iteration = itertools.count()
  while np.linalg.norm(old_u - u) > tolerance:
    old_u = u
    u = 1.0 / np.dot(big_k_tilde, (distn_b / np.dot(big_k.T, u)))  # was matmul
    logging.debug('Sinkhorn iteration: %d', next(iteration))

  v = distn_b / np.dot(big_k.T, u)

  coupling = np.dot(np.dot(np.diag(u), big_k), np.diag(v))  # was matmul

  return u, v, coupling


def wasserstein_coupling_on_the_real_line(x1, x2):
  """Compute the Wasserstein coupling between two sets of real numbers.

  Args:
    x1: A vector of real numbers.
    x2: A vector of real numbers. Need not be the same length as x1.

  Returns:
    The Wasserstein coupling.
  """
  # If x1 and x2 were sorted, then the coupling matrix would only depend on
  # their lengths. The sort order of x1 and x2 only require us to permute the
  # coupling matrix to match. So, we first compute the coupling matrix for a
  # sorted x1, x2:
  l1 = len(x1)
  l2 = len(x2)

  # Shortcut 1: the coupling matrix for sorted vectors of the same length is the
  # identity matrix, scaled.
  if l1 == l2:
    return sparse.spdiags([np.ones_like(x1) / l1], [0], l1, l1)

  # Shortcut 2: we may have cached the sorted coupling matrix that we're about
  # to compute.
  elif (l1, l2) in _WCOTRL_CACHE:
    coupling = _WCOTRL_CACHE[(l1, l2)]

  # Alas, we need to compute the coupling matrix after all.
  else:
    # coupling = np.zeros((l1, l2))

    data = []
    rows = []
    cols = []

    i, j = 0, 0
    while i < l1 and j < l2:
      logging.debug('WCOTRL: %d,%d', i, j)

      il2 = i * l2
      jl1 = j * l1

      il2_limit = il2 + l2 - 1
      jl1_limit = jl1 + l1 - 1

      intersection_size = max(
          0,
          1 + min(il2_limit, jl1_limit) - max(il2, jl1)
      )
      # coupling[i, j] = intersection_size
      data.append(intersection_size)
      rows.append(i)
      cols.append(j)

      if il2_limit <= jl1_limit: i += 1
      if il2_limit >= jl1_limit: j += 1

    coupling = sparse.csr_matrix((data, (rows, cols)), shape=(l1, l2))
    _WCOTRL_CACHE[(l1, l2)] = coupling  # Cache for later.

  # Now we permute the rows and columns of the coupling matrix to match the sort
  # orders of x1 and x2.
  row_inds = np.argsort(np.argsort(x1))  # ⍋⍋x1
  col_inds = np.argsort(np.argsort(x2))
  coupling = coupling[row_inds, :]
  coupling = coupling[:, col_inds]

  return coupling / (l1 * l2)
