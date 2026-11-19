"""Finite checks for the degree-one obstruction to exact free norm bounds for permutation matrices.

For N = 2, ..., 5 and every ordered pair (A, B) of N x N permutation matrices this script
  1. builds the witness of Lemma 3.1 (a unit vector v orthogonal to the constants with Cv = zv and
     |Re <v, Dv>| >= 1/n, where {C, D} = {A, B} and n = N - 1),
  2. checks that the polynomial P of Theorem 1.1 has Rayleigh quotient at least 2 + 1/n^4 at v,
  3. checks that some member of the finite family of Corollary 1.2 has norm on the orthogonal
     complement of the constants at least 2 + 1/n^4,
and in exact rational arithmetic it checks the free upper bound 2 + 3/(4 n^4) and the example of
Remark 3.2. Requires NumPy. Run: python check_obstruction.py
"""
from fractions import Fraction as Fr
from itertools import permutations
import cmath
import numpy as np

TOL = 1e-9


def perm_matrix(p):
    N = len(p)
    M = np.zeros((N, N))
    for x, y in enumerate(p):
        M[y, x] = 1.0  # e_x -> e_{p(x)}
    return M


def complement_basis(N):
    # orthonormal basis of the orthogonal complement of the constants
    Q, _ = np.linalg.qr(np.column_stack([np.ones(N)] + [np.eye(N)[:, k] for k in range(N - 1)]))
    return Q[:, 1:]


def witness(A, B, a, b, N):
    """Return (C, D, z, v) as in Lemma 3.1."""
    n = N - 1
    for C, D, c in ((A, B, a), (B, A, b)):
        fixed = [x for x in range(N) if c[x] == x]
        if fixed:
            x = fixed[0]
            v = -np.ones(N) / N
            v[x] += 1.0
            return C, D, 1.0, v / np.linalg.norm(v)
    E = complement_basis(N)
    w, V = np.linalg.eig(E.T @ A @ E)
    # orthonormalise inside each eigenspace (A restricted to H is unitary, so eigenspaces are orthogonal)
    best = None
    for lam in np.unique(np.round(w, 9)):
        idx = np.where(np.abs(w - lam) < 1e-7)[0]
        Qs, _ = np.linalg.qr(V[:, idx])
        for k in range(Qs.shape[1]):
            u = E @ Qs[:, k]
            val = np.real(np.vdot(u, B @ u))
            if best is None or val < best[0]:
                best = (val, lam, u)
    assert best[0] <= -1.0 / n + TOL, (a, b, best[0])
    return A, B, complex(best[1]), best[2]


def P_matrix(C, D, z, sigma, t):
    return np.conj(z) * C + z * C.T + sigma * t * (D + D.T)


def norm_on_H(M, N):
    E = complement_basis(N)
    return np.max(np.abs(np.linalg.eigvalsh(E.T @ M @ E)))


def roots_of_unity(N):
    return [cmath.exp(2j * cmath.pi * j / k) for k in range(1, N + 1) for j in range(k)]


for N in range(2, 6):
    n = N - 1
    t = 1.0 / (2 * n ** 3)
    target = 2 + 1.0 / n ** 4
    # exact free bound with s = 1/n^2: 2 + s^2/4 + 2 t^2/s = 2 + 3/(4 n^4)
    s_, t_ = Fr(1, n * n), Fr(1, 2 * n ** 3)
    assert 2 + s_ ** 2 / 4 + 2 * t_ ** 2 / s_ == 2 + Fr(3, 4 * n ** 4)
    fam = [(i, z, sg) for i in (0, 1) for z in roots_of_unity(N) for sg in (1, -1)]
    perms = list(permutations(range(N)))
    worst_lemma, worst_family, pairs = np.inf, np.inf, 0
    for a in perms:
        A = perm_matrix(a)
        for b in perms:
            B = perm_matrix(b)
            C, D, z, v = witness(A, B, a, b, N)
            re = np.real(np.vdot(v, D @ v))
            assert abs(np.sum(v)) < TOL and np.linalg.norm(C @ v - z * v) < 1e-7 and abs(re) >= 1.0 / n - TOL
            sigma = 1 if re > 0 else -1
            P = P_matrix(C, D, z, sigma, t)
            rq = np.real(np.vdot(v, P @ v))
            worst_lemma = min(worst_lemma, rq - target)
            best = max(norm_on_H(P_matrix(*(A, B) if i == 0 else (B, A), z, sg, t), N)
                       for i, z, sg in fam)
            worst_family = min(worst_family, best - target)
            pairs += 1
    assert worst_lemma >= -TOL and worst_family >= -TOL
    print(f"N={N}: {pairs} ordered pairs, family size {len(fam)}; "
          f"min Rayleigh - (2+1/n^4) = {worst_lemma:.3e}, min family norm - (2+1/n^4) = {worst_family:.3e}; "
          f"free bound 2+3/(4n^4) = {2 + 3 / (4 * n ** 4):.6f}")

# Remark 3.2: N = 3, C = (1 2), D = (0 1 2), t = 1/16, P = C + C^* - t (D + D^*)
C = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
D = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
t = Fr(1, 16)
P = [[Fr(C[i][j] + C[j][i]) - t * (D[i][j] + D[j][i]) for j in range(3)] for i in range(3)]
for vec, lam in (([2, -1, -1], Fr(33, 16)), ([0, 1, -1], Fr(-31, 16))):
    assert [sum(P[i][j] * vec[j] for j in range(3)) for i in range(3)] == [lam * x for x in vec]
s = Fr(1, 4)
free = 2 + s ** 2 / 4 + 2 * t ** 2 / s
assert free == Fr(131, 64) and Fr(33, 16) - free == Fr(1, 64)
print("N=3 example: eigenvalues 33/16 and -31/16 on the complement, free bound 131/64, gap 1/64")
print("all checks pass")
