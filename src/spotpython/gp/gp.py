import numpy as np


class GP:
    def __init__(self, m, n, X, Z, d, g, dK):
        self.m = m
        self.n = n
        self.X = np.copy(X)
        self.Z = np.copy(Z)
        self.d = d
        self.g = g
        self.K = None
        self.Ki = None
        self.ldetK = None
        self.KiZ = None
        self.dK = None
        self.d2K = None
        self.F = 0
        if dK:
            self.newdKGP()

    def buildGP(self, dK):
        assert self.K is None
        if self.d == 0:
            assert not dK  # sanity check for degenerate GP

        # Build covariance matrix
        self.K = np.zeros((self.n, self.n))
        if self.d > 0:
            self.K = self.covar_symm(self.X, self.d, self.g)
        else:
            np.fill_diagonal(self.K, 1)

        # Invert covariance matrix
        self.Ki = np.eye(self.n)
        if self.d > 0:
            Kchol = np.linalg.cholesky(self.K)
            self.Ki = np.linalg.inv(Kchol).T @ np.linalg.inv(Kchol)
            self.ldetK = 2 * np.sum(np.log(np.diag(Kchol)))
        else:
            self.ldetK = 0.0

        # phi <- t(Z) %*% Ki %*% Z
        self.calc_ZtKiZ()

        # Calculate derivatives and Fisher info based on them
        if dK:
            self.newdKGP()

    def covar_symm(self, X, d, g):
        # Implement the covariance function here
        pass

    def calc_ZtKiZ(self):
        # Implement the calculation of ZtKiZ here
        pass

    def newdKGP(self):
        # Implement the calculation of derivatives and Fisher info here
        pass


def newGP(m, n, X, Z, d, g, dK):
    gp = GP(m, n, X, Z, d, g, dK)
    gp.buildGP(dK)
    return gp


def newGP_R(m_in, n_in, X_in, Z_in, d_in, g_in, dK_in):
    X = np.array(X_in).reshape((n_in, m_in))
    gp = newGP(m_in, n_in, X, Z_in, d_in, g_in, dK_in)
    return gp
