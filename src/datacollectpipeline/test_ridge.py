# Save as test_ridge.py
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from dataclasses import dataclass
@dataclass
class MockConfig:
    prediction_horizon: int = 5
    test_size: float = 0.2
    alpha_1: float = 1e-6
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6
class MockPredictor:
    def __init__(self, config):
        self.config = config
        self.bayesian_models = {}
    def train_bayesian_ridge(self, X, y_multi):
        r2_scores = []
        for h in range(1, self.config.prediction_horizon + 1):
            y_h = y_multi[:, h-1]
            X_train, X_test, y_train, y_test = train_test_split(X, y_h, test_size=self.config.test_size, shuffle=False)
            model_h = BayesianRidge(alpha_1=self.config.alpha_1, alpha_2=self.config.alpha_2,
                                    lambda_1=self.config.lambda_1, lambda_2=self.config.lambda_2)
            model_h.fit(X_train, y_train)
            y_pred = model_h.predict(X_test)
            r2_h = r2_score(y_test, y_pred)
            r2_scores.append(r2_h)
            self.bayesian_models[h] = model_h
        return self.bayesian_models, r2_scores
X = np.random.rand(200, 29)
y = np.random.rand(200, 5)
predictor = MockPredictor(MockConfig())
result = predictor.train_bayesian_ridge(X, y)
models, scores = result
print(f"Models: {len(models)}, Scores: {len(scores)}, Avg R²: {np.mean(scores):.3f}")
