from __future__ import annotations

import copy
import random
import re
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.base import ProbabilisticClassifier, ensure_probability_matrix, to_numpy
from preprocessing.feature_engineering import frame_to_padded_stage_sequence, frame_to_stage_sequence


class SoftF1Loss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.softmax(logits, dim=1)[:, 1]
        target = target.float()
        tp = (prob * target).sum()
        fp = (prob * (1.0 - target)).sum()
        fn = ((1.0 - prob) * target).sum()
        soft_f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
        return 1.0 - soft_f1


class TorchBinaryClassifier(ProbabilisticClassifier):
    def __init__(
        self,
        stage: int | None = None,
        random_state: int = 42,
        epochs: int = 50,
        batch_size: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        validation_fraction: float = 0.2,
        device: str = "auto",
        loss: str = "ce",
        patience: int = 8,
        use_class_weight: bool = False,
        **_,
    ):
        self.stage = stage
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.loss = loss
        self.patience = patience
        self.use_class_weight = use_class_weight
        self.device = self._resolve_device(device)
        self.scaler: StandardScaler | None = None
        self.network_: nn.Module | None = None

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _set_seed(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _raw_features(self, X) -> np.ndarray:
        return to_numpy(X)

    def _prepare_features(self, X, fit: bool) -> np.ndarray:
        raw = self._raw_features(X).astype(np.float32)
        original_shape = raw.shape
        flat = raw.reshape(raw.shape[0], -1)
        if fit or self.scaler is None:
            self.scaler = StandardScaler().fit(flat)
        scaled = self.scaler.transform(flat).astype(np.float32)
        return scaled.reshape(original_shape)

    @abstractmethod
    def _build_network(self, input_shape: tuple[int, ...]) -> nn.Module:
        raise NotImplementedError

    def _criterion(self, y_train: np.ndarray) -> nn.Module:
        if self.loss == "soft_f1":
            return SoftF1Loss()
        if self.loss == "ce" or not self.use_class_weight:
            return nn.CrossEntropyLoss()
        counts = np.bincount(y_train.astype(int), minlength=2).astype(np.float32)
        weights = counts.sum() / np.maximum(2.0 * counts, 1.0)
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=self.device))

    def fit(self, X, y):
        self._set_seed()
        features = self._prepare_features(X, fit=True)
        labels = np.asarray(y, dtype=np.int64)
        if self.validation_fraction and 0.0 < self.validation_fraction < 1.0:
            X_train, X_val, y_train, y_val = train_test_split(
                features,
                labels,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=labels,
            )
        else:
            X_train, X_val, y_train, y_val = features, None, labels, None
        self.network_ = self._build_network(tuple(features.shape[1:])).to(self.device)
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.epochs, 1))
        criterion = self._criterion(y_train)

        dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        best_state = None
        best_loss = float("inf")
        bad_epochs = 0
        for _epoch in range(self.epochs):
            self.network_.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.network_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            if X_val is not None:
                current_loss = self._validation_loss(X_val, y_val, criterion)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_state = copy.deepcopy(self.network_.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.patience:
                        break
        if best_state is not None:
            self.network_.load_state_dict(best_state)
        return self

    def _validation_loss(self, X_val: np.ndarray, y_val: np.ndarray, criterion: nn.Module) -> float:
        assert self.network_ is not None
        self.network_.eval()
        with torch.no_grad():
            x = torch.from_numpy(X_val).to(self.device)
            y = torch.from_numpy(y_val).to(self.device)
            return float(criterion(self.network_(x), y).detach().cpu().item())

    def predict_proba(self, X) -> np.ndarray:
        if self.network_ is None:
            raise RuntimeError("model is not fitted")
        features = self._prepare_features(X, fit=False)
        self.network_.eval()
        batches = []
        dataset = TensorDataset(torch.from_numpy(features))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for (batch_x,) in loader:
                logits = self.network_(batch_x.to(self.device))
                batches.append(torch.softmax(logits, dim=1).cpu().numpy())
        return ensure_probability_matrix(np.vstack(batches))


class StageSequenceMixin:
    def _raw_features(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.stage is None:
                raise ValueError("stage is required for DataFrame sequence conversion")
            return frame_to_stage_sequence(X, self.stage)
        array = np.asarray(X, dtype=np.float32)
        if array.ndim == 2:
            return array[:, None, :]
        return array


class SharedStageSequenceMixin:
    def _raw_features(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return frame_to_padded_stage_sequence(X, stage=self.stage)
        array = np.asarray(X, dtype=np.float32)
        if array.ndim == 2:
            if self.stage is None:
                return array[:, None, :]
            sequence = np.zeros((array.shape[0], 4, array.shape[1] + 1), dtype=np.float32)
            sequence[:, self.stage - 1, :-1] = array
            sequence[:, : self.stage, -1] = 1.0
            return sequence
        return array

    def _prepare_features(self, X, fit: bool) -> np.ndarray:
        raw = self._raw_features(X).astype(np.float32)
        if raw.ndim != 3:
            return super()._prepare_features(raw, fit=fit)
        values = raw[:, :, :-1]
        visibility_mask = raw[:, :, -1:]
        original_shape = values.shape
        flat = values.reshape(values.shape[0], -1)
        if fit or self.scaler is None:
            self.scaler = StandardScaler().fit(flat)
        scaled = self.scaler.transform(flat).astype(np.float32).reshape(original_shape)
        return np.concatenate([scaled, visibility_mask.astype(np.float32)], axis=-1)


class MetaProbabilitySequenceMixin:
    def _raw_features(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            legacy_pattern = re.compile(r"(.+)_p_s([1-4])$")
            parsed = []
            for column in X.columns:
                match = legacy_pattern.match(column)
                if match:
                    parsed.append((column, match.group(1), int(match.group(2))))
            if parsed:
                learners = []
                for _column, learner, _stage in parsed:
                    if learner not in learners:
                        learners.append(learner)
                stages = sorted({stage for _column, _learner, stage in parsed})
                frames = []
                for stage in stages:
                    cols = [f"{learner}_p_s{stage}" for learner in learners]
                    missing = [col for col in cols if col not in X.columns]
                    if missing:
                        raise ValueError(f"missing legacy meta probability columns: {missing}")
                    frames.append(X.loc[:, cols].to_numpy(dtype=np.float32))
                return np.stack(frames, axis=1)
            probability_cols = [column for column in X.columns if column.endswith("_prob")]
            array = X.loc[:, probability_cols].to_numpy(dtype=np.float32) if probability_cols else X.to_numpy(dtype=np.float32)
        else:
            array = np.asarray(X, dtype=np.float32)
        if array.ndim == 2:
            array = array[:, :, None]
        return array
