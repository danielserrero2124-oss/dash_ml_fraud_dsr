"""Modelo original: RVM-Fraude, una Relevance Vector Machine de clasificación implementada desde cero.

Base bibliográfica (indexada en Scopus y Web of Science):
  * Tipping, M. E. (2001). Sparse Bayesian learning and the relevance vector machine.
    Journal of Machine Learning Research, 1, 211-244.
  * Tipping, M. E. y Faul, A. C. (2003). Fast marginal likelihood maximisation for sparse Bayesian
    models. Proceedings of AISTATS 2003.
  * MacKay, D. J. C. (1992). The evidence framework applied to classification networks.
    Neural Computation, 4(5), 720-736. https://doi.org/10.1162/neco.1992.4.5.720

Formulación (clasificación binaria, y ∈ {0, 1}):

    Verosimilitud   p(y | w) = Π_n σ(a_n)^{y_n} (1 − σ(a_n))^{1 − y_n},   a_n = wᵀ φ(x_n)
    Prior ARD       p(w | α) = Π_i N(w_i | 0, α_i⁻¹)      (una precisión α_i por función base)

1. Paso de Laplace: con α fijo se busca el máximo a posteriori w* con Newton (IRLS):
       g = Φᵀ(y − p) − A w,     H = Φᵀ B Φ + A,     w ← w + H⁻¹ g      (B = diag(p(1 − p)), A = diag(α))
   y la posterior se aproxima por una gaussiana N(w*, Σ) con Σ = H⁻¹.
2. Paso de evidencia (máxima verosimilitud tipo II): γ_i = 1 − α_i Σ_ii,   α_i ← γ_i / w*_i².
3. Las funciones base con α_i → ∞ tienen peso exactamente 0 y se eliminan: el modelo se vuelve
   disperso y conserva solo los "vectores relevantes" (regularización automática, navaja de Occam).

La predicción usa la aproximación moderada de MacKay, que incorpora la incertidumbre de w:
       P(y = 1 | x) ≈ σ( μ_a / sqrt(1 + π s² / 8) ),   μ_a = w*ᵀ φ(x),   s² = φ(x)ᵀ Σ φ(x)

Las funciones base φ(x) las construye el Pipeline (splines cúbicos para las variables continuas y
one-hot para las discretas y categóricas), así que el modelo es aditivo y cada variable tiene una
curva de contribución con su banda de incertidumbre bayesiana.

No se importa ningún modelo: solo numpy para el álgebra y la interfaz de scikit-learn.
"""

import numpy as np  # Álgebra lineal (Newton, inversa del hessiano, determinantes)
from sklearn.base import BaseEstimator, ClassifierMixin  # Interfaz de scikit-learn (clone, GridSearchCV, Pipeline)


def _sigmoide(z):  # Función logística σ(z) = 1 / (1 + e^(−z))
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))  # El recorte evita desbordamientos en exp()


class RVMClassifier(BaseEstimator, ClassifierMixin):
    """Relevance Vector Machine para clasificación binaria (Tipping, 2001).

    Parámetros
    ----------
    max_iter : iteraciones máximas del ciclo Laplace → evidencia.
    tol : cambio relativo mínimo de log α para seguir iterando.
    alpha_inicial : precisión inicial de todas las funciones base.
    umbral_poda : α a partir del cual una función base se considera irrelevante y se elimina.
    iter_newton : iteraciones máximas de Newton (IRLS) en cada paso de Laplace.
    prediccion_moderada : True usa la salida moderada de MacKay (incluye la incertidumbre de w).
    """

    def __init__(self, max_iter=200, tol=1e-3, alpha_inicial=1.0, umbral_poda=1e6, iter_newton=25, prediccion_moderada=True):
        self.max_iter = max_iter  # Guarda cada hiperparámetro sin modificarlo (requisito de scikit-learn)
        self.tol = tol  # Tolerancia de convergencia
        self.alpha_inicial = alpha_inicial  # Precisión inicial
        self.umbral_poda = umbral_poda  # Umbral de poda
        self.iter_newton = iter_newton  # Iteraciones de Newton
        self.prediccion_moderada = prediccion_moderada  # Tipo de predicción

    def _laplace(self, P, y, a, w):  # Paso 1: máximo a posteriori de w con Newton y aproximación gaussiana
        for _ in range(self.iter_newton):  # Iteraciones de Newton (IRLS)
            p = _sigmoide(P @ w)  # Probabilidades actuales
            g = P.T @ (y - p) - a * w  # Gradiente del logaritmo de la posterior
            H = (P * (p * (1 - p))[:, None]).T @ P + np.diag(a + 1e-8)  # Hessiano negativo: ΦᵀBΦ + A (más un pequeño jitter numérico)
            paso = np.linalg.solve(H, g)  # Dirección de Newton H⁻¹ g
            w = w + paso  # Actualiza los pesos
            if np.max(np.abs(paso)) < 1e-6:  # Convergió
                break  # Sale del bucle de Newton
        p = _sigmoide(P @ w)  # Probabilidades en el óptimo
        H = (P * (p * (1 - p))[:, None]).T @ P + np.diag(a + 1e-8)  # Hessiano en el óptimo
        return w, H, p  # Pesos MAP, hessiano y probabilidades

    def _log_evidencia(self, y, p, w, a, H):  # Logaritmo de la evidencia aproximada por Laplace (diagnóstico)
        log_ver = float(np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))  # ln p(y | w*)
        log_prior = 0.5 * float(np.sum(np.log(a))) - 0.5 * float(np.sum(a * w ** 2))  # ln p(w* | α) sin la constante 2π
        _, logdet_H = np.linalg.slogdet(H)  # ln |H| = −ln |Σ|
        return log_ver + log_prior - 0.5 * float(logdet_H)  # ln p(y | α) ≈ ln p(y|w*) + ln p(w*|α) + ½ ln |Σ| (las constantes 2π se cancelan)

    def fit(self, X, y):  # Entrena la RVM
        X = np.asarray(X, dtype=float)  # Matriz de funciones base (ya transformada por el Pipeline)
        y = np.asarray(y, dtype=float)  # Objetivo 0/1
        self.classes_ = np.array([0, 1])  # Clases conocidas (requisito de ClassifierMixin)
        Phi = np.hstack([np.ones((len(X), 1)), X])  # Agrega la función base constante (sesgo) en la columna 0
        M = Phi.shape[1]  # Número total de funciones base
        alpha = np.full(M, float(self.alpha_inicial))  # Precisiones iniciales
        alpha[0] = 1e-6  # El sesgo casi no se regulariza
        w = np.zeros(M)  # Pesos iniciales
        activas = np.arange(M)  # Funciones base todavía relevantes
        self.historial_ = []  # Diagnóstico por iteración: relevantes y log-evidencia
        for it in range(self.max_iter):  # Ciclo Laplace → evidencia
            P, a = Phi[:, activas], alpha[activas]  # Submatriz y precisiones de las funciones activas
            wa, H, p = self._laplace(P, y, a, w[activas])  # Paso de Laplace
            Sigma = np.linalg.inv(H)  # Covarianza posterior Σ = H⁻¹
            self.historial_.append({"iteracion": it + 1, "relevantes": int(len(activas)),  # Guarda el estado actual
                                    "log_evidencia": self._log_evidencia(y, p, wa, a, H)})
            gamma = 1 - a * np.diag(Sigma)  # Grado en que los datos determinan cada peso (0 = lo fija el prior, 1 = los datos)
            nuevo = np.clip(gamma, 1e-12, None) / (wa ** 2 + 1e-12)  # Regla de MacKay: α_i = γ_i / w_i²
            nuevo[activas == 0] = 1e-6  # El sesgo se mantiene sin regularizar
            cambio = np.max(np.abs(np.log(nuevo) - np.log(a)))  # Cambio máximo de log α
            w[:] = 0.0  # Las funciones podadas quedan con peso cero
            w[activas] = wa  # Pesos de las funciones activas
            alpha[activas] = np.minimum(nuevo, 1e12)  # Actualiza las precisiones
            activas = activas[alpha[activas] < self.umbral_poda]  # Poda: elimina las funciones con α enorme (peso ≈ 0)
            if cambio < self.tol:  # Las precisiones ya no cambian
                break  # Convergencia del ciclo
        P, a = Phi[:, activas], alpha[activas]  # Estado final
        wa, H, p = self._laplace(P, y, a, w[activas])  # Último paso de Laplace con el conjunto definitivo
        self.relevantes_ = activas  # Índices de los vectores relevantes (0 = sesgo)
        self.mu_ = wa  # Media posterior de los pesos relevantes
        self.Sigma_ = np.linalg.inv(H)  # Covarianza posterior de los pesos relevantes
        self.alpha_ = alpha  # Precisiones finales de todas las funciones base (para graficar la poda)
        self.n_bases_ = M  # Número de funciones base iniciales
        self.n_iter_ = len(self.historial_)  # Iteraciones realizadas
        return self  # Convención de scikit-learn

    def _phi(self, X):  # Funciones base relevantes de nuevas filas
        X = np.asarray(X, dtype=float)  # Matriz de entrada
        return np.hstack([np.ones((len(X), 1)), X])[:, self.relevantes_]  # Agrega el sesgo y conserva solo las relevantes

    def decision_function(self, X):  # Logit medio μ_a = w*ᵀ φ(x)
        return self._phi(X) @ self.mu_  # Producto con los pesos posteriores

    def predict_proba(self, X):  # Probabilidades de ambas clases
        P = self._phi(X)  # Funciones base relevantes
        mu_a = P @ self.mu_  # Media del logit
        if self.prediccion_moderada:  # Salida moderada de MacKay
            s2 = np.einsum("ij,jk,ik->i", P, self.Sigma_, P)  # Varianza del logit φᵀ Σ φ para cada fila
            mu_a = mu_a / np.sqrt(1 + np.pi * s2 / 8)  # Encoge el logit cuando la predicción es incierta
        p = _sigmoide(mu_a)  # Probabilidad de fraude
        return np.column_stack([1 - p, p])  # Columna 0: legítima; columna 1: fraude

    def predict(self, X):  # Clase con umbral 0,5 (el umbral de negocio se elige aparte, en validación)
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)  # 1 si la probabilidad supera 0,5
