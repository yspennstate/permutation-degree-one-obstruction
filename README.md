# Exact free norm bounds fail for permutation matrices at degree one

Yitzchak Shmalo

For every size N ≥ 2, every r ≥ 2 and every r-tuple of N × N permutation matrices, there is a self-adjoint
noncommutative polynomial of degree one, with scalar coefficients, whose norm on the orthogonal complement of the
constant vector exceeds its norm in the left regular representation of the free group by at least 1/(4(N−1)^4).
The polynomial can be taken from an explicit set of at most 2N(N+1) polynomials that depends only on N. This
answers in the negative the question of van Handel (arXiv:2507.00346, Section 6.5) on a Ramanujan form of strong
convergence for polynomials of degree at most q(N).

## Files

- `main.tex`, `main.pdf`: the paper.
- `check_obstruction.py`: for N = 2, ..., 5 and every ordered pair of N × N permutation matrices, builds the test
  vector of Lemma 3.1, checks the Rayleigh quotient bound of Theorem 1.1, and checks that some member of the set
  of Corollary 1.2 has norm at least 2 + 1/(N−1)^4 on the complement of the constants. This is the left inequality
  of Theorem 1.1; the free norm is not computed (the script checks only the rational identity used for the upper
  bound in the proof). It also checks the N = 3 example in exact rational arithmetic. Requires Python 3 and NumPy;
  runs in about five minutes.

```
python check_obstruction.py
```
