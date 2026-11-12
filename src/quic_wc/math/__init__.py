"""Differentiable building blocks: Cayley bases, compound matrices, structured transforms."""

from .cayley import cayley, cayley_from_params, inverse_cayley, params_from_skew, skew_from_params, skew_size
from .compound import combinations, compound_dim, compound_matrix, compound_matrix_reference
from .orthogonal import BlockLayout, QuICOrthogonal, make_permutation, select_columns

__all__ = [
    "cayley", "cayley_from_params", "inverse_cayley", "params_from_skew", "skew_from_params", "skew_size",
    "combinations", "compound_dim", "compound_matrix", "compound_matrix_reference",
    "BlockLayout", "QuICOrthogonal", "make_permutation", "select_columns",
]
