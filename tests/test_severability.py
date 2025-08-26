import numpy as np
import pytest

from severability import severability


class ER:
    def __init__(self, n, p, seed):
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.p = p

    def sample(self):
        # sample from Bernoulli(p)
        U = self.rng.uniform(size=(self.n, self.n))
        A_unsymmetric = np.asarray(U <= self.p, dtype=int)

        # make symmetric by copying upper triangular part
        A = np.tril(A_unsymmetric) + np.tril(A_unsymmetric).T

        # remove diagonal
        np.fill_diagonal(A, 0)

        return A


@pytest.fixture
def adj():
    er = ER(n=200, p=0.1, seed=1)
    return er.sample()


def test_transition_matrix(adj):
    N = adj.shape[0]
    d = adj @ np.ones(N)
    D_inv = np.diag(1 / d)
    P = D_inv @ adj
    np.testing.assert_allclose(P, severability.transition_matrix(np.matrix(adj)))


def test_retention(adj):
    P = severability.transition_matrix(np.matrix(adj))
    r = severability.retention(P)
    assert r == 1.0


def test_mixing(P):
    pass


def test_severability_of_matrix_power(P):
    pass


def test_severability_of_component(P, C, t):
    pass


def test_component_cover(P, t, max_size):
    pass


def test_node_component(P, i, t, max_size):
    pass


def test_connected_component(P, C):
    pass


def test_component_optimise(P, C, t, max_size):
    pass


def test_greedy_add_step(P, C, t):
    pass


def test_greedy_remove_step(P, C, t):
    pass


def test_kernighan_lin_step(P, C, t):
    pass
