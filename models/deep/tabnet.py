from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.base import OptionalDependencyError, ProbabilisticClassifier, ensure_probability_matrix, to_numpy


class TabNetModel(ProbabilisticClassifier):
    def __init__(
        self,
        stage: int | None = None,
        random_state: int = 42,
        n_d: int = 16,
        n_a: int = 16,
        n_steps: int = 3,
        gamma: float = 1.5,
        max_epochs: int = 20,
        batch_size: int = 1024,
        patience: int = 8,
        virtual_batch_size: int = 256,
        **kwargs,
    ):
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError as exc:
            raise OptionalDependencyError("Install pytorch-tabnet to use TabNetModel.") from exc
        self.model = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            optimizer_params=kwargs.pop("optimizer_params", dict(lr=2e-2)),
            seed=random_state,
            verbose=0,
            **kwargs,
        )
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.virtual_batch_size = virtual_batch_size
        self.random_state = random_state
        self.scaler: StandardScaler | None = None

    def fit(self, X, y):
        x = to_numpy(X)
        y = np.asarray(y, dtype=np.int64)
        self.scaler = StandardScaler().fit(x)
        x = self.scaler.transform(x).astype(np.float32)
        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y,
        )
        self.model.fit(
            X_train=x_train,
            y_train=y_train,
            eval_set=[(x_val, y_val)],
            eval_name=["val"],
            eval_metric=["logloss"],
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            patience=self.patience,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        x = to_numpy(X)
        if self.scaler is not None:
            x = self.scaler.transform(x).astype(np.float32)
        return ensure_probability_matrix(self.model.predict_proba(x))
