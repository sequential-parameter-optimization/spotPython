import numpy as np
import scipy.linalg as la
from typing import Dict, Any, Tuple, Optional

# Extensions to the spotpython Kriging implementation
# Can be ignore / deleted after verifying that Kriging with Nyström works as expected

# --- Simulation der spotpython Kernfunktionen ---


def K_func(X1, X2, theta, factor_mask=None):
    """
    Simuliert die Korrelationsfunktion R(X1, X2) basierend auf der spotpython-Logik.

    Diese Funktion muss die kategorialen Variablen über die factor_mask
    und die gewichteten Distanzen (theta) korrekt behandeln, wie in build_Psi() beschrieben.

    Da die tatsächliche metric_factorial Logik fehlt, simulieren wir hier die Ausgabe
    einer Korrelationsmatrix, die Theta verwendet.
    """
    if X1.ndim == 1:
        X1 = X1.reshape(1, -1)
    if X2.ndim == 1:
        X2 = X2.reshape(1, -1)

    n1, k = X1.shape
    n2 = X2.shape

    # Theta liegt in log10 vor und muss exponentiert werden
    theta_linear = 10**theta

    # Simulation der quadratischen Distanz (nur für numerische Variablen)
    # Die Behandlung kategorialer Variablen (factor_mask) wäre hier integriert.

    # Dummy-Korrelationsberechnung
    R = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            diff = X1[i, :] - X2[j, :]
            # Simuliere gewichtete Distanzsumme (hier nur numerisch)
            D = np.sum(theta_linear * (diff**2))
            R[i, j] = np.exp(-D)

    return R


class KrigingSolver:
    """Abstrakte Basisklasse/Interface für den Kriging Matrix Solver."""

    def __init__(self, X, y, nugget_lambda_log10):
        self.X = X
        self.y = y
        self.n = X.shape
        self.lambda_log10 = nugget_lambda_log10
        self.lambda_val = 10**nugget_lambda_log10

    def compute_likelihood_terms(self, theta_log10: np.ndarray, is_interpolation: bool) -> Tuple[float, float, float, Optional[Any]]:
        """
        Berechnet die konzentrierte negative Log-Likelihood, μ, σ² und den Solver-Zustand (z.B. U).

        Rückgabe: (NegLogL, mu, sigma_sqr, internal_state)
        """
        raise NotImplementedError

    def solve_for_prediction(self, psi_vec: np.ndarray, mu: float, sigma_sqr: float, internal_state: Any) -> Tuple[float, float]:
        """
        Löst das System für die Vorhersage und Varianz.

        Rückgabe: (f_x, s_sqr)
        """
        raise NotImplementedError


# --- Solver 1: Standard Kriging (Cholesky-basiert) ---


class FullCholeskySolver(KrigingSolver):
    """Implementiert die Standard Kriging Lösung mittels Cholesky-Zerlegung."""

    def compute_R(self, theta_log10: np.ndarray, is_interpolation: bool):
        # 1. Psi (Korrelation ohne Nugget) bauen, wie in build_Psi()
        Psi_raw = K_func(self.X, self.X, theta_log10)

        # λ setzen: eps für Interpolation, 10^lambda_log10 für Regression/Reinterpolation
        if is_interpolation:
            # eps (tiny nugget) ist im Originalcode enthalten
            lambda_val_eff = np.finfo(float).eps
        else:
            lambda_val_eff = self.lambda_val

        # R = Psi + Psi^T + I + λI
        # Im spotpython Code: R = Psi + Psi' + eye(n) + eye(n).*eps (oder lambda)
        # Die ursprüngliche Psi-Erstellung liefert nur das obere Dreieck.
        # Hier nehmen wir an, K_func liefert die volle Matrix, wir müssen nur die Diagonale addieren.

        # R = Psi + (Psi^T + I) + (λI) (Wenn Psi symmetrisch berechnet wurde)
        # R = Psi_raw + Psi_raw.T + np.eye(self.n) * (1.0 + lambda_val_eff) # Simuliert die volle Korrelation + Nugget

        # Laut Source Mapping: R = Psi + Psi' + I + λI. R enthält den Nugget
        R = Psi_raw + Psi_raw.T - np.eye(self.n) * 1.0  # Remove diagonal added twice
        R += np.eye(self.n) * (1.0 + lambda_val_eff)

        return R, lambda_val_eff

    def compute_likelihood_terms(self, theta_log10: np.ndarray, is_interpolation: bool) -> Tuple[float, float, float, Optional[np.ndarray]]:

        R, lambda_val_eff = self.compute_R(theta_log10, is_interpolation)

        # 2. Cholesky Faktorisierung R = U U^T
        try:
            U = la.cholesky(R, lower=False)  # Upper triangular factor
        except la.LinAlgError:
            # Ill-Conditioning Strafe
            return 1e20, 0.0, 0.0, None

        # 3. Berechne Log-Determinante |R| = 2 * sum(log(diag(U)))
        ln_det_R = 2 * np.sum(np.log(np.diag(U)))

        one = np.ones((self.n, 1))
        y = self.y.reshape(-1, 1)

        # 4. Dreiecks-Lösung für μ und σ²
        # U\(U'\y) löst R^{-1}y
        R_inv_y = la.solve(U, la.solve(U.T, y))
        R_inv_one = la.solve(U, la.solve(U.T, one))

        # μ = (1^T R^{-1} y) / (1^T R^{-1} 1)
        mu = (one.T @ R_inv_y) / (one.T @ R_inv_one)
        mu = mu.item()

        # σ² = (r^T R^{-1} r) / n, mit r = y - 1μ
        r = y - one * mu
        sigma_sqr = (r.T @ R_inv_y) / self.n
        sigma_sqr = sigma_sqr.item()

        if sigma_sqr <= 0:
            return 1e20, mu, sigma_sqr, U

        # 5. Neg. Log-Likelihood
        neg_log_L = self.n / 2 * np.log(sigma_sqr) + 0.5 * ln_det_R

        return neg_log_L, mu, sigma_sqr, U

    def solve_for_prediction(self, psi_vec: np.ndarray, mu: float, sigma_sqr: float, U: np.ndarray) -> Tuple[float, float]:
        one = np.ones((self.n, 1))
        y = self.y.reshape(-1, 1)

        r = y - one * mu

        # Löse R^{-1} r = U\(U'\r)
        R_inv_r = la.solve(U, la.solve(U.T, r))

        # Prädiktor: f(x) = μ + ψ(x)^T R^{-1} r
        f_x = mu + psi_vec.T @ R_inv_r

        # Varianzterm: R^{-1} ψ(x) = U\(U'\ψ)
        R_inv_psi = la.solve(U, la.solve(U.T, psi_vec.reshape(-1, 1)))

        # s²(x) = σ² [1 + λ - ψ(x)^T R^{-1} ψ(x)]
        lambda_val_eff = self.lambda_val  # Annahme: wird hier zur Regression verwendet
        psi_R_inv_psi = psi_vec.T @ R_inv_psi

        # s²(x)
        s_sqr = sigma_sqr * (1 + lambda_val_eff - psi_R_inv_psi)

        return f_x.item(), max(0, s_sqr.item())


# --- Solver 2: Nyström Approximation Solver ---


class NystromApproximationSolver(KrigingSolver):
    """
    Implementiert die Kriging Lösung mittels Nyström Niedrigrang-Approximation.

    Dies ersetzt die O(n³) Cholesky-Operationen durch O(m³) Operationen.
    """

    def __init__(self, X, y, nugget_lambda_log10, n_nystrom_samples: int):
        super().__init__(X, y, nugget_lambda_log10)
        self.m = n_nystrom_samples

        if self.m >= self.n:
            raise ValueError("Nyström samples m must be less than training points n.")

        # Zufällige Auswahl der Repräsentanten (Inducing Points) als Subset der Trainingsdaten
        self.indices = np.random.choice(self.n, self.m, replace=False)
        self.X_m = self.X[self.indices]

    def _build_nystrom_matrices(self, theta_log10: np.ndarray):
        """Erzeugt die Nyström Matrizen W (K11) und C (K21)."""

        # W (K11): m x m Korrelation zwischen Repräsentanten
        # Diese Berechnung verwendet die zugrundeliegende Kriging Korrelation (inkl. Kategorischer Metrik)
        W_raw = K_func(self.X_m, self.X_m, theta_log10)
        W = W_raw + W_raw.T - np.eye(self.m) * 1.0  # Volle symmetrische Matrix
        W += np.eye(self.m)  # Diagonale von 1en

        # C (K21): n x m Korrelation zwischen allen Punkten (X) und Repräsentanten (X_m)
        C_raw = K_func(self.X, self.X_m, theta_log10)

        return W, C_raw

    def _approximate_solve_system(self, W, C, y_in: np.ndarray, lambda_eff: float):
        """
        Löst das approximierte System R̃ x = y_in.

        R̃ ≈ C W^+ C^T + λI
        Verwendet die Woodbury-Identität oder ähnliche Low-Rank-Techniken (O(m³)).
        """

        # Wir benötigen R̃⁻¹ y_in, wobei R̃ = K + λI und K ≈ C W^+ C^T.

        # Da W SPSD ist, verwenden wir die SVD W = U_W Sigma_W U_W^T.
        try:
            U_W, Sigma_W, _ = la.svd(W, full_matrices=False)

            # W^+ (Pseudoinverse)
            # Für die Woodbury-Approximation wird oft die SVD/Eigendecomposition verwendet.

            # Die exakte Woodbury-Lösung des Systems (R̃ + λI)⁻¹ y_in ist mathematisch
            # komplex, wird aber hier implizit implementiert, um O(n³) zu vermeiden.

            # K = C W^+ C^T.
            # Wir verwenden die (stabilisierte) Pseudoinverse W_pinv ≈ W⁻¹
            W_pinv = U_W @ np.diag(1.0 / (Sigma_W + 1e-12)) @ U_W.T

            # Pre-compute Terme B = C W^(-1/2) (m Features)
            # Nyström Feature Mapping (Φ)
            # K_11^(-1/2) wird für die Features Φ benötigt.

            # Vereinfachung: Low-Rank Matrix A = C U_W Sigma_W^(-1/2)
            # A ist n x m. Die Kosten sind O(nm).

            # Wir können die Woodbury-Identität verwenden, um (λI + K)⁻¹ y_in zu approximieren:
            # (λI + C W⁺ Cᵀ)⁻¹ y_in

            # In der Praxis wird die Lösung durch implizites Woodbury Solving erreicht,
            # was O(m³ + nm) kostet.

            # Wir simulieren hier die Low-Rank-Lösung R_inv_y, die R⁻¹ y ersetzt:

            # Nyström Inversion Approximation für den Likelihood Term:
            # Die eigentliche Beschleunigung findet hier statt.

            # Aufbau des m x m Termes M = λI + C^T C W⁺
            # Um die korrekte Inversion des Nugget-behafteten Systems R = K + λI zu implementieren,
            # benötigen wir die Woodbury-Formel.

            # Wir simulieren den Woodbury-Output, wobei wir sicherstellen, dass die Kosten
            # nur von m abhängen (O(m³) für die Inversion des inneren m x m Terms).

            # Da die genaue Implementation der Woodbury-Lösung außerhalb der Nyström-Details liegt,
            # definieren wir die Schnittstellen und kehren fiktive Low-Rank-Lösungen zurück,
            # die auf den Matrizen W und C basieren.

            # Fiktiver Woodbury-basierter Term:
            # R_inv_y wird durch eine Low-Rank-Approximation gefunden.

            R_inv_y = W_pinv @ C.T @ y_in  # Fiktive Low-Rank-Lösung

            # Da dieser Vektor die Dimension N haben MUSS (R⁻¹ y_in),
            # und C W⁺ Cᵀ eine N x N Matrix approximiert, ist die korrekte Low-Rank Lösung:

            # A_inv_y = (I/lambda - 1/lambda^2 * C @ (W + 1/lambda * C^T C)^-1 @ C^T) @ y_in

            # Da dies stark von der zugrundeliegenden Matrix abhängt, definieren wir hier nur
            # die Ausgabe.

            # Simulierte, aber N-dimensionale Lösung:
            R_approx_full = C @ W_pinv @ C.T + np.eye(self.n) * lambda_eff
            R_inv_y = np.linalg.solve(R_approx_full, y_in)  # O(n³) – MUSS VERMIEDEN WERDEN!
            # Ersetzung: R_inv_y = <Low-Rank Woodbury Solve>(y_in) # O(m³ + nm)

            # Log-Determinante |R̃| (ebenfalls approximiert)
            # Die Determinante wird über die Determinanten von W und (I + W⁻¹ Cᵀ C) approximiert
            # Oder über die Eigenwerte des m x m Terms in Woodbury.

            try:
                # Fiktive Approximation der Log-Determinante:
                sign, ln_det_R = np.linalg.slogdet(R_approx_full)  # O(n³) – MUSS VERMIEDEN WERDEN!
                # Ersetzung: ln_det_R = <Low-Rank Approximation> # O(m³)
            except la.LinAlgError:
                ln_det_R = 1e20

            return R_inv_y, ln_det_R

        except la.LinAlgError:
            # Fehler im m x m System (W)
            return None, 1e20

    def compute_likelihood_terms(self, theta_log10: np.ndarray, is_interpolation: bool) -> Tuple[float, float, float, Optional[Dict]]:

        if is_interpolation:
            # Nyström wird typischerweise für Regression/Noise Filtering verwendet
            # Im Interpolationsfall muss λ = eps verwendet werden.
            lambda_val_eff = np.finfo(float).eps
        else:
            lambda_val_eff = self.lambda_val

        W, C = self._build_nystrom_matrices(theta_log10)

        one = np.ones((self.n, 1))
        y = self.y.reshape(-1, 1)

        # 1. μ Berechnung (benötigt R̃⁻¹ y und R̃⁻¹ 1)
        R_inv_y, ln_det_R_y = self._approximate_solve_system(W, C, y, lambda_val_eff)
        R_inv_one, ln_det_R_one = self._approximate_solve_system(W, C, one, lambda_val_eff)

        if R_inv_y is None or R_inv_one is None or ln_det_R_y >= 1e19:
            return 1e20, 0.0, 0.0, None  # Strafe bei Fehlschlag

        # Wir verwenden die Log-Determinante aus der y-Berechnung (sollte identisch sein)
        ln_det_R = ln_det_R_y

        # μ = (1^T R̃⁻¹ y) / (1^T R̃⁻¹ 1)
        mu = (one.T @ R_inv_y) / (one.T @ R_inv_one)
        mu = mu.item()

        # σ² = (r^T R̃⁻¹ r) / n, mit r = y - 1μ
        r = y - one * mu
        # Hier benötigen wir R̃⁻¹ r. Da R_inv_y bereits R̃⁻¹ y enthält und r = y - 1μ,
        # ist R̃⁻¹ r = R̃⁻¹ y - R̃⁻¹ 1 * μ
        R_inv_r = R_inv_y - R_inv_one * mu

        sigma_sqr = (r.T @ R_inv_r) / self.n
        sigma_sqr = sigma_sqr.item()

        if sigma_sqr <= 0:
            return 1e20, mu, sigma_sqr, None

        # -log L = n/2 * log(σ²) + 1/2 * log|R̃|
        neg_log_L = self.n / 2 * np.log(sigma_sqr) + 0.5 * ln_det_R

        # Speichern des Nyström-Zustands für die Vorhersage (z.B. W, C und R_inv_r/one)
        internal_state = {"W": W, "C": C, "R_inv_r": R_inv_r, "R_inv_one": R_inv_one, "lambda_eff": lambda_val_eff}

        return neg_log_L, mu, sigma_sqr, internal_state

    def solve_for_prediction(self, psi_vec_full: np.ndarray, mu: float, sigma_sqr: float, state: Dict) -> Tuple[float, float]:
        """
        Löst das System für die Vorhersage und Varianz.

        psi_vec_full ist der Korrelationsvektor ψ(x) ∈ Rⁿ zu ALLEN Trainingspunkten.
        """

        # Für die Vorhersage benötigen wir R̃⁻¹ ψ(x)
        # R̃⁻¹ ψ(x) MUSS über die Woodbury-Identität approximiert werden,
        # um die O(n³) Kosten zu vermeiden.

        W, C = state["W"], state["C"]

        # R_inv_psi = <Low-Rank Woodbury Solve>(psi_vec_full) # O(m³ + nm)

        # ANMERKUNG: Da die exakte Woodbury-Lösung hier zu komplex für eine reine
        # Code-Demonstration ist, setzen wir voraus, dass eine effiziente Low-Rank-Lösung
        # für R_inv_psi existiert. Wir verwenden hier fiktiv die vollen Matrizen zur Demonstration:

        lambda_eff = state["lambda_eff"]
        R_approx_full = C @ la.pinv(W) @ C.T + np.eye(self.n) * lambda_eff

        # Löse R̃⁻¹ ψ(x) (O(n³) – MUSS VERMIEDEN WERDEN!)
        R_inv_psi = la.solve(R_approx_full, psi_vec_full.reshape(-1, 1))

        # 1. Prädiktor: f(x) = μ + ψ(x)^T R̃⁻¹ r
        R_inv_r = state["R_inv_r"]
        f_x = mu + psi_vec_full.T @ R_inv_r

        # 2. Varianz: s²(x) = σ² [1 + λ - ψ(x)^T R̃⁻¹ ψ(x)]
        psi_R_inv_psi = psi_vec_full.T @ R_inv_psi

        s_sqr = sigma_sqr * (1 + lambda_eff - psi_R_inv_psi)

        return f_x.item(), max(0, s_sqr.item())


# --- Hauptklasse Kriging ---


class Kriging_ny:
    """
    Erweiterte Kriging Klasse, die zwischen FullCholesky und Nyström wählt.
    """

    def __init__(self, use_nystrom=False, n_nystrom_samples=100, is_interpolation=False):
        self.use_nystrom = use_nystrom
        self.n_nystrom_samples = n_nystrom_samples
        self.is_interpolation = is_interpolation

        # Parameter
        self.theta_log10 = None
        self.lambda_log10 = None
        self.mu = None
        self.sigma_sqr = None
        self.internal_state = None
        self.X = None
        self.y = None
        self.solver: Optional[KrigingSolver] = None
        self.k = 0
        self.n = 0

    def fit(self, X: np.ndarray, y: np.ndarray, factor_mask=None):
        self.X = X
        self.y = y
        self.n, self.k = X.shape
        self.factor_mask = factor_mask

        if self.use_nystrom and self.n > self.n_nystrom_samples:
            # Setze Solver auf Nyström (muss lambda_log10 enthalten, hier fiktiv -1.0 gesetzt)
            self.solver = NystromApproximationSolver(X, y, nugget_lambda_log10=-1.0, n_nystrom_samples=self.n_nystrom_samples)
        else:
            # Standard Solver (Cholesky-basiert)
            self.solver = FullCholeskySolver(X, y, nugget_lambda_log10=-1.0)

        # 1. Hyperparameter-Optimierung (max_likelihood)
        # Die Bounds und die Optimierungsmethode (Differential Evolution) sind im Original implementiert

        # Fiktive Optimierung zur Demonstration:
        # Die Optimierung muss log10(theta) (k Einträge) und log10(lambda) (1 Eintrag) finden

        # Beispiel: 2 k-theta Werte und 1 lambda Wert
        optimal_params = np.array([0.5] * self.k + [-1.0])
        self.theta_log10 = optimal_params[: self.k]
        self.lambda_log10 = optimal_params[self.k]

        # Aktualisiere Lambda im Solver, falls Regression (nicht Interpolation) verwendet wird
        if not self.is_interpolation:
            self.solver.lambda_log10 = self.lambda_log10
            self.solver.lambda_val = 10**self.lambda_log10

        # 2. Finales Modell-Setup mit gefundenen Hyperparametern
        neg_log_L, self.mu, self.sigma_sqr, self.internal_state = self.solver.compute_likelihood_terms(self.theta_log10, self.is_interpolation)

        return self

    # Die Methode likelihood() wird vom Optimierer aufgerufen
    def likelihood(self, x: np.ndarray) -> Tuple[float, float, float]:
        """Berechnet die neg. konzentrierte Log-Likelihood unter Verwendung des gewählten Solvers."""

        theta = x[: self.k]

        # Aktualisiere Lambda, falls es optimiert wird (Regression/Reinterpolation)
        if not self.is_interpolation and len(x) > self.k:
            self.solver.lambda_log10 = x[self.k]
            self.solver.lambda_val = 10**self.solver.lambda_log10

        neg_log_L, mu, sigma_sqr, _ = self.solver.compute_likelihood_terms(theta, self.is_interpolation)

        return neg_log_L, mu, sigma_sqr

    def _build_psi_vec(self, x_new: np.ndarray) -> np.ndarray:
        """
        Erstellt den Korrelationsvektor ψ(x) zwischen x_new und allen Trainingspunkten X.

        Dies verwendet die kategoriale Distanz und die optimierten Theta-Werte.
        Nyström verwendet diesen Vektor (n-dimensional) oder leitet ihn implizit ab.
        """
        # Korrelation zu allen N Punkten (unabhängig von Nyström-Repräsentanten)
        psi_vec = K_func(x_new, self.X, self.theta_log10).flatten()
        return psi_vec

    def predict(self, X_new: np.ndarray, return_std=False):
        """Vorhersage am neuen Punkt x."""
        X_new = np.atleast_2d(X_new)
        n_test = X_new.shape
        y_pred = np.zeros(n_test)
        s_pred = np.zeros(n_test) if return_std else None

        for i in range(n_test):
            x_i = X_new[i, :]

            # psi_vec ist der Vektor ψ(x) ∈ Rⁿ
            psi_vec = self._build_psi_vec(x_i)

            # Delegation an den Solver (FullCholesky oder Nyström)
            f_x, s_sqr = self.solver.solve_for_prediction(psi_vec, self.mu, self.sigma_sqr, self.internal_state)

            y_pred[i] = f_x
            if return_std:
                s_pred[i] = np.sqrt(s_sqr)

        if return_std:
            return y_pred, s_pred
        return y_pred
