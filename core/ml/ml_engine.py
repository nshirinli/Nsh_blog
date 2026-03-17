"""Machine-Learning engine for ChemEng Platform.

Wraps scikit-learn behind a stable API used by MLController.
All public functions are stateless; state lives in the controller.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional scikit-learn import – fail gracefully
# ---------------------------------------------------------------------------
try:
    from sklearn.linear_model import (
        LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression,
    )
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor,
        RandomForestClassifier, GradientBoostingClassifier,
    )
    from sklearn.svm import SVR, SVC
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA as _PCA
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder,
    )
    from sklearn.model_selection import (
        train_test_split, cross_val_score, learning_curve as _lc,
    )
    from sklearn.metrics import (
        r2_score, mean_squared_error, mean_absolute_error,
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
        silhouette_score,
    )
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_sklearn() -> None:
    if not SKLEARN_OK:
        raise ImportError(
            "scikit-learn is required for the ML module.\n"
            "Install it with:  pip install scikit-learn"
        )


def _scaler_from_name(name: str):
    if not SKLEARN_OK:
        return None
    return {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "None": None,
    }.get(name)


def _regression_model(name: str, **kwargs):
    _require_sklearn()
    models: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=kwargs.get("alpha", 1.0)),
        "Lasso": Lasso(alpha=kwargs.get("alpha", 0.1), max_iter=10_000),
        "ElasticNet": ElasticNet(alpha=kwargs.get("alpha", 0.1), l1_ratio=0.5, max_iter=10_000),
        "Random Forest": RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", None) or None,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 5),
            random_state=42,
        ),
        "SVR": SVR(C=kwargs.get("C", 1.0), epsilon=0.1),
        "KNN": KNeighborsRegressor(n_neighbors=kwargs.get("n_neighbors", 5)),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=kwargs.get("max_depth", None) or None,
            random_state=42,
        ),
    }
    m = models.get(name)
    if m is None:
        raise ValueError(f"Unknown regression model: {name!r}")
    return m


def _classification_model(name: str, **kwargs):
    _require_sklearn()
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            C=kwargs.get("C", 1.0), max_iter=1_000, random_state=42,
        ),
        "SVM (SVC)": SVC(C=kwargs.get("C", 1.0), probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=kwargs.get("n_neighbors", 5)),
        "Random Forest": RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", None) or None,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 5),
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=kwargs.get("max_depth", None) or None,
            random_state=42,
        ),
    }
    m = models.get(name)
    if m is None:
        raise ValueError(f"Unknown classification model: {name!r}")
    return m


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV or Excel (xlsx / xls) into a DataFrame."""
    p = Path(filepath)
    ext = p.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(p)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .csv or .xlsx.")


def data_summary(df: pd.DataFrame) -> dict:
    """Return shape, dtypes, missing counts, and describe() dict."""
    num = df.select_dtypes(include=[np.number])
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "describe": num.describe().to_dict(),
    }


def preprocess(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    missing_strategy: str = "drop",
    scaler_name: str = "StandardScaler",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Preprocess dataframe → train/test split with optional scaling."""
    _require_sklearn()

    sub = df[feature_cols + [target_col]].copy()

    if missing_strategy == "drop":
        sub = sub.dropna()
    elif missing_strategy == "mean":
        sub = sub.fillna(sub.mean(numeric_only=True))
    elif missing_strategy == "median":
        sub = sub.fillna(sub.median(numeric_only=True))
    elif missing_strategy == "zero":
        sub = sub.fillna(0)

    X = sub[feature_cols].values.astype(float)
    y = sub[target_col].values

    label_encoder = None
    if y.dtype.kind in ("U", "O", "S"):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y).astype(int)
    else:
        y = y.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = _scaler_from_name(scaler_name)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler, "label_encoder": label_encoder,
        "n_samples": len(sub),
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def train_regression(
    X_train, y_train, X_test, y_test,
    model_name: str = "Random Forest",
    **kwargs,
) -> dict:
    """Train a regression model; return metrics + prediction arrays."""
    _require_sklearn()
    model = _regression_model(model_name, **kwargs)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    fi = None
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
    elif hasattr(model, "coef_"):
        fi = np.abs(np.atleast_1d(model.coef_)).ravel()

    return {
        "model": model,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
        "rmse_train": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "mae_test": float(mean_absolute_error(y_test, y_pred_test)),
        "feature_importances": fi,
    }


def train_classification(
    X_train, y_train, X_test, y_test,
    model_name: str = "Random Forest",
    **kwargs,
) -> dict:
    """Train a classification model; return metrics + confusion matrix."""
    _require_sklearn()
    model = _classification_model(model_name, **kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    fi = None
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
    elif hasattr(model, "coef_"):
        c = model.coef_
        fi = np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c).ravel()

    return {
        "model": model,
        "y_pred": y_pred,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "classes": np.unique(np.concatenate([y_train, y_test])),
        "feature_importances": fi,
    }


def run_clustering(
    X: np.ndarray,
    algorithm: str = "KMeans",
    n_clusters: int = 3,
    eps: float = 0.5,
    min_samples: int = 5,
) -> dict:
    """Cluster data; return labels, silhouette score, and elbow data."""
    _require_sklearn()
    elbow_k = elbow_inertia = centers = None

    if algorithm == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = model.fit_predict(X)
        centers = model.cluster_centers_
        k_range = range(2, min(11, len(X)))
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            km.fit(X)
            inertias.append(km.inertia_)
        elbow_k = list(k_range)
        elbow_inertia = inertias
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
    elif algorithm == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm!r}")

    n_found = len(np.unique(labels[labels >= 0]))
    sil = None
    if n_found > 1 and len(X) > n_found:
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            pass

    return {
        "labels": labels,
        "centers": centers,
        "n_clusters_found": n_found,
        "silhouette": sil,
        "elbow_k": elbow_k,
        "elbow_inertia": elbow_inertia,
    }


def run_pca(
    X: np.ndarray,
    n_components: int,
    feature_names: list[str] | None = None,
) -> dict:
    """Run PCA; return scores, loadings, explained variance."""
    _require_sklearn()
    n_components = min(n_components, X.shape[1], X.shape[0])
    pca = _PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    return {
        "pca": pca,
        "scores": scores,
        "explained_variance_ratio": evr,
        "cumulative_variance": np.cumsum(evr),
        "loadings": pca.components_,
        "feature_names": feature_names or [f"f{i}" for i in range(X.shape[1])],
    }


def run_cross_validation(
    X, y,
    model_name: str,
    task: str = "regression",
    cv_folds: int = 5,
) -> dict:
    """k-fold CV + learning curve for the chosen model."""
    _require_sklearn()
    if task == "regression":
        model = _regression_model(model_name)
        scoring = "r2"
    else:
        model = _classification_model(model_name)
        scoring = "accuracy"

    scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)

    n_points = min(8, len(X))
    train_sizes_rel = np.linspace(0.1, 1.0, n_points)
    train_sizes, train_sc, val_sc = _lc(
        model, X, y,
        cv=min(cv_folds, 5),
        train_sizes=train_sizes_rel,
        scoring=scoring,
    )

    return {
        "scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "cv_folds": cv_folds,
        "scoring": scoring,
        "train_sizes": train_sizes,
        "train_scores_mean": train_sc.mean(axis=1),
        "val_scores_mean": val_sc.mean(axis=1),
        "train_scores_std": train_sc.std(axis=1),
        "val_scores_std": val_sc.std(axis=1),
    }


def predict_new(model, scaler, X_new: np.ndarray) -> np.ndarray:
    """Scale X_new (if scaler present) then predict."""
    if scaler is not None:
        X_new = scaler.transform(X_new)
    return model.predict(X_new)
