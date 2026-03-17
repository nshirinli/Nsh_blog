"""ML controller — maintains model/data state and formats results for the UI."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.ml.ml_engine import (
    SKLEARN_OK,
    load_data,
    data_summary,
    preprocess,
    train_regression,
    train_classification,
    run_clustering,
    run_pca,
    run_cross_validation,
    predict_new,
)


class MLController:
    """Stateful controller for the ML page.

    Internal state
    --------------
    _df            : loaded DataFrame
    _X_train/test  : preprocessed feature arrays
    _y_train/test  : target arrays
    _scaler        : fitted scaler (or None)
    _label_encoder : fitted LabelEncoder (or None)
    _model         : last trained model
    _feature_cols  : list of feature column names
    _target_col    : target column name
    _task          : 'regression' or 'classification'
    """

    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None
        self._X_train = self._X_test = None
        self._y_train = self._y_test = None
        self._scaler = None
        self._label_encoder = None
        self._model = None
        self._feature_cols: list[str] = []
        self._target_col: str = ""
        self._task: str = "regression"

    # ------------------------------------------------------------------
    # Tab 1 — Data Import
    # ------------------------------------------------------------------

    def load_data(self, filepath: str) -> tuple[str, dict]:
        try:
            self._df = load_data(filepath)
        except Exception as exc:
            return f"Error loading file:\n{exc}", {}

        info = data_summary(self._df)
        rows, cols = info["shape"]
        missing_total = sum(info["missing"].values())

        lines = [
            f"File loaded: {Path(filepath).name}",
            f"Shape      : {rows} rows × {cols} columns",
            f"Missing    : {missing_total} total",
            "",
            "Column name" + " " * 22 + "dtype" + " " * 12 + "missing",
            "─" * 60,
        ]
        for col, dtype in info["dtypes"].items():
            m = info["missing"].get(col, 0)
            lines.append(f"  {col[:30]:30s}  {dtype:15s}  {m}")

        lines += ["", "Numeric statistics:", "─" * 60]
        try:
            lines.append(self._df.describe().to_string())
        except Exception:
            pass

        return "\n".join(lines), {"info": info, "df": self._df}

    def get_columns(self) -> list[str]:
        if self._df is None:
            return []
        return list(self._df.columns)

    def get_numeric_columns(self) -> list[str]:
        if self._df is None:
            return []
        return list(self._df.select_dtypes(include=[np.number]).columns)

    def get_preview_data(self, max_rows: int = 200) -> tuple[list[str], list[list]]:
        """Return (headers, rows) for QTableWidget population."""
        if self._df is None:
            return [], []
        sub = self._df.head(max_rows)
        return list(sub.columns), sub.values.tolist()

    # ------------------------------------------------------------------
    # Tab 2 — Preprocessing
    # ------------------------------------------------------------------

    def preprocess(
        self,
        target_col: str,
        feature_cols: list[str],
        missing_strategy: str,
        scaler_name: str,
        test_size: float,
    ) -> tuple[str, dict]:
        if self._df is None:
            return "No data loaded. Import a file first.", {}
        if not SKLEARN_OK:
            return "scikit-learn not installed.\nRun: pip install scikit-learn", {}
        if not feature_cols:
            return "Select at least one feature column.", {}

        self._target_col = target_col
        self._feature_cols = feature_cols

        try:
            r = preprocess(
                self._df, target_col, feature_cols,
                missing_strategy, scaler_name, test_size,
            )
        except Exception as exc:
            return f"Preprocessing error:\n{exc}", {}

        self._X_train = r["X_train"]
        self._X_test = r["X_test"]
        self._y_train = r["y_train"]
        self._y_test = r["y_test"]
        self._scaler = r["scaler"]
        self._label_encoder = r["label_encoder"]

        # Infer task type
        unique_y = len(np.unique(self._y_train))
        is_int_target = self._y_train.dtype.kind in ("i", "u")
        self._task = "classification" if (unique_y <= 20 and is_int_target) else "regression"

        lines = [
            "Preprocessing complete",
            "═" * 44,
            f"Target column   : {target_col}",
            f"Feature columns : {', '.join(feature_cols)}",
            f"Missing values  : {missing_strategy}",
            f"Scaler          : {scaler_name}",
            f"Test fraction   : {test_size:.0%}",
            "",
            f"Total samples   : {r['n_samples']}",
            f"Training set    : {r['n_train']} samples",
            f"Test set        : {r['n_test']} samples",
            f"Features        : {r['n_features']}",
            "",
            f"Detected task   : {self._task.capitalize()}",
            f"  (unique target values: {unique_y})",
            "",
            "Ready to train a model.",
        ]
        return "\n".join(lines), {"X_train": self._X_train, "X_test": self._X_test}

    # ------------------------------------------------------------------
    # Tab 3 — Regression
    # ------------------------------------------------------------------

    def train_regression(self, model_name: str, **kwargs) -> tuple[str, dict]:
        if self._X_train is None:
            return "Run preprocessing first (Tab 2).", {}
        if not SKLEARN_OK:
            return "scikit-learn not installed.\nRun: pip install scikit-learn", {}

        try:
            r = train_regression(
                self._X_train, self._y_train,
                self._X_test, self._y_test,
                model_name, **kwargs,
            )
        except Exception as exc:
            return f"Training error:\n{exc}", {}

        self._model = r["model"]

        lines = [
            f"Regression Model : {model_name}",
            "═" * 44,
            "Training set metrics:",
            f"  R²    = {r['r2_train']:.4f}",
            f"  RMSE  = {r['rmse_train']:.4g}",
            "",
            "Test set metrics:",
            f"  R²    = {r['r2_test']:.4f}",
            f"  RMSE  = {r['rmse_test']:.4g}",
            f"  MAE   = {r['mae_test']:.4g}",
        ]

        fi = r["feature_importances"]
        if fi is not None and len(fi) > 0:
            lines += ["", "Feature importances (top 10):"]
            sorted_idx = np.argsort(fi)[::-1][:10]
            for idx in sorted_idx:
                name = self._feature_cols[idx] if idx < len(self._feature_cols) else f"f{idx}"
                lines.append(f"  {name[:30]:30s}  {fi[idx]:.4f}")

        return "\n".join(lines), {
            "type": "regression",
            "y_test": self._y_test,
            "y_pred": r["y_pred_test"],
            "feature_importances": fi,
            "feature_names": list(self._feature_cols),
            "model_name": model_name,
            "r2_test": r["r2_test"],
        }

    # ------------------------------------------------------------------
    # Tab 4 — Classification
    # ------------------------------------------------------------------

    def train_classification(self, model_name: str, **kwargs) -> tuple[str, dict]:
        if self._X_train is None:
            return "Run preprocessing first (Tab 2).", {}
        if not SKLEARN_OK:
            return "scikit-learn not installed.\nRun: pip install scikit-learn", {}

        try:
            r = train_classification(
                self._X_train, self._y_train,
                self._X_test, self._y_test,
                model_name, **kwargs,
            )
        except Exception as exc:
            return f"Training error:\n{exc}", {}

        self._model = r["model"]

        lines = [
            f"Classification Model : {model_name}",
            "═" * 44,
            f"Accuracy   : {r['accuracy']:.4f}",
            f"Precision  : {r['precision']:.4f}  (weighted avg)",
            f"Recall     : {r['recall']:.4f}  (weighted avg)",
            f"F1 Score   : {r['f1']:.4f}  (weighted avg)",
            "",
            "Classification Report:",
            "─" * 44,
            r["classification_report"],
        ]

        return "\n".join(lines), {
            "type": "classification",
            "confusion_matrix": r["confusion_matrix"],
            "classes": r["classes"],
            "feature_importances": r["feature_importances"],
            "feature_names": list(self._feature_cols),
            "model_name": model_name,
            "accuracy": r["accuracy"],
        }

    # ------------------------------------------------------------------
    # Tab 5 — Clustering
    # ------------------------------------------------------------------

    def run_clustering(
        self,
        algorithm: str,
        n_clusters: int,
        feature_cols: list[str],
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> tuple[str, dict]:
        if self._df is None:
            return "No data loaded.", {}
        if not SKLEARN_OK:
            return "scikit-learn not installed.\nRun: pip install scikit-learn", {}

        try:
            from sklearn.preprocessing import StandardScaler as _SS
            sub = self._df[feature_cols].dropna()
            X_scaled = _SS().fit_transform(sub.values.astype(float))
            r = run_clustering(X_scaled, algorithm, n_clusters, eps, min_samples)
        except Exception as exc:
            return f"Clustering error:\n{exc}", {}

        sil_str = f"{r['silhouette']:.4f}" if r["silhouette"] is not None else "N/A"
        lines = [
            f"Clustering : {algorithm}",
            "═" * 44,
            f"Clusters found   : {r['n_clusters_found']}",
            f"Silhouette score : {sil_str}",
            "",
            "Cluster sizes:",
        ]
        for lbl in sorted(set(r["labels"])):
            count = int((r["labels"] == lbl).sum())
            name = f"Cluster {lbl}" if lbl >= 0 else "Noise (DBSCAN)"
            lines.append(f"  {name}: {count} samples")

        return "\n".join(lines), {
            "type": "clustering",
            "X": X_scaled,
            "labels": r["labels"],
            "elbow_k": r["elbow_k"],
            "elbow_inertia": r["elbow_inertia"],
            "algorithm": algorithm,
            "feature_names": feature_cols,
        }

    # ------------------------------------------------------------------
    # Tab 6 — PCA
    # ------------------------------------------------------------------

    def run_pca(self, n_components: int, feature_cols: list[str]) -> tuple[str, dict]:
        if self._df is None:
            return "No data loaded.", {}
        if not SKLEARN_OK:
            return "scikit-learn not installed.\nRun: pip install scikit-learn", {}

        try:
            from sklearn.preprocessing import StandardScaler as _SS
            sub = self._df[feature_cols].dropna()
            X_scaled = _SS().fit_transform(sub.values.astype(float))
            r = run_pca(X_scaled, n_components, feature_cols)
        except Exception as exc:
            return f"PCA error:\n{exc}", {}

        evr = r["explained_variance_ratio"]
        lines = [
            f"PCA  ({n_components} components requested)",
            "═" * 44,
            f"Input features     : {len(feature_cols)}",
            f"Components returned: {len(evr)}",
            "",
            "Explained Variance:",
            "─" * 44,
        ]
        cumul = 0.0
        for i, ev in enumerate(evr):
            cumul += ev
            lines.append(f"  PC{i+1:2d}: {ev*100:6.2f}%  (cumulative: {cumul*100:6.2f}%)")

        return "\n".join(lines), {
            "type": "pca",
            "scores": r["scores"],
            "explained_variance_ratio": evr,
            "cumulative_variance": r["cumulative_variance"],
            "loadings": r["loadings"],
            "feature_names": feature_cols,
        }

    # ------------------------------------------------------------------
    # Tab 7 — Cross-Validation
    # ------------------------------------------------------------------

    def run_cross_validation(
        self,
        model_name: str,
        task: str,
        cv_folds: int,
    ) -> tuple[str, dict]:
        if self._X_train is None:
            return "Run preprocessing first (Tab 2).", {}
        if not SKLEARN_OK:
            return "scikit-learn not installed.\nRun: pip install scikit-learn", {}

        X = np.vstack([self._X_train, self._X_test])
        y = np.concatenate([self._y_train, self._y_test])

        try:
            r = run_cross_validation(X, y, model_name, task, cv_folds)
        except Exception as exc:
            return f"Cross-validation error:\n{exc}", {}

        score_strs = ", ".join(f"{s:.4f}" for s in r["scores"])
        lines = [
            f"Cross-Validation : {model_name}",
            "═" * 44,
            f"Task    : {task.capitalize()}",
            f"Folds   : {cv_folds}",
            f"Metric  : {r['scoring']}",
            "",
            f"Fold scores  : {score_strs}",
            "",
            f"Mean  ± Std  : {r['mean']:.4f} ± {r['std']:.4f}",
            f"95% CI       : [{r['mean'] - 2*r['std']:.4f},  {r['mean'] + 2*r['std']:.4f}]",
        ]
        return "\n".join(lines), {
            "type": "cv",
            "scores": r["scores"],
            "train_sizes": r["train_sizes"],
            "train_scores_mean": r["train_scores_mean"],
            "val_scores_mean": r["val_scores_mean"],
            "train_scores_std": r["train_scores_std"],
            "val_scores_std": r["val_scores_std"],
            "scoring": r["scoring"],
            "model_name": model_name,
        }

    # ------------------------------------------------------------------
    # Tab 8 — Predict
    # ------------------------------------------------------------------

    def predict(self, values: list[float]) -> tuple[str, dict]:
        if self._model is None:
            return "No trained model. Train a regression or classification model first.", {}

        X_new = np.array(values, dtype=float).reshape(1, -1)
        try:
            y_pred = predict_new(self._model, self._scaler, X_new)
        except Exception as exc:
            return f"Prediction error:\n{exc}", {}

        pred_val = y_pred[0]
        if self._label_encoder is not None:
            try:
                pred_label = self._label_encoder.inverse_transform([int(pred_val)])[0]
                pred_str = f"{pred_label}  (class {int(pred_val)})"
            except Exception:
                pred_str = str(pred_val)
        else:
            pred_str = f"{pred_val:.6g}"

        lines = [
            "Prediction",
            "═" * 44,
            f"Model : {type(self._model).__name__}",
            "",
            "Input values:",
        ]
        for name, val in zip(self._feature_cols, values):
            lines.append(f"  {name[:30]:30s} = {val:.6g}")
        lines += ["", f"Predicted value : {pred_str}"]

        return "\n".join(lines), {"prediction": float(pred_val)}

    # ------------------------------------------------------------------
    # Helpers queried by the page
    # ------------------------------------------------------------------

    def get_feature_names(self) -> list[str]:
        return list(self._feature_cols)

    def is_data_loaded(self) -> bool:
        return self._df is not None

    def is_preprocessed(self) -> bool:
        return self._X_train is not None

    def is_model_trained(self) -> bool:
        return self._model is not None
