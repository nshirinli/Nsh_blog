import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tempfile

st.set_page_config(layout="wide", page_title="Machine Learning - ChemEng")
st.title("🤖 Machine Learning")

if "ml_ctrl" not in st.session_state:
    from app.controllers.ml_controller import MLController
    st.session_state.ml_ctrl = MLController()
ctrl = st.session_state.ml_ctrl

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Data Import", "Preprocessing", "Regression", "Classification",
    "Clustering", "PCA", "Cross-Validation", "Predict"
])

# Tab 1 - Data Import
with tab1:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        st.subheader("Upload Dataset")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        if uploaded_file is not None:
            suffix = ".csv" if uploaded_file.name.endswith(".csv") else ".xlsx"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                ctrl.load_data(tmp_path)
                st.success(f"Loaded: {uploaded_file.name}")
                st.session_state["ml_data_loaded"] = True
            except Exception as e:
                st.error(f"Error loading file: {e}")
    with col_out:
        if ctrl.is_data_loaded():
            try:
                preview = ctrl.get_preview_data()
                st.subheader("Data Preview")
                st.dataframe(preview, use_container_width=True)
            except Exception as e:
                st.error(f"Preview error: {e}")
        else:
            st.info("No data loaded yet. Upload a file on the left.")

# Tab 2 - Preprocessing
with tab2:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        if ctrl.is_data_loaded():
            with st.form("form_t2"):
                st.subheader("Preprocessing")
                try:
                    columns = ctrl.get_columns()
                except Exception:
                    columns = []
                target_col = st.selectbox("Target column", columns)
                feature_cols = st.multiselect("Feature columns", columns,
                                              default=[c for c in columns if c != target_col])
                missing_strategy = st.selectbox("Missing values strategy", ["mean", "median", "most_frequent", "drop"])
                scaler_name = st.selectbox("Scaler", ["none", "standard", "minmax", "robust"])
                test_size = st.slider("Test size fraction", 0.1, 0.5, 0.2)
                submitted = st.form_submit_button("Apply Preprocessing", use_container_width=True)
            if submitted:
                try:
                    ctrl.preprocess(target_col, feature_cols, missing_strategy, scaler_name, test_size)
                    st.session_state["ml_target"] = target_col
                    st.session_state["ml_features"] = feature_cols
                    st.success("Preprocessing complete.")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please load data first (Data Import tab).")
    with col_out:
        if ctrl.is_data_loaded():
            st.info("Configure preprocessing settings and click Apply.")

# Tab 3 - Regression
with tab3:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        if ctrl.is_preprocessed():
            with st.form("form_t3"):
                st.subheader("Regression")
                reg_model = st.selectbox("Model", [
                    "LinearRegression", "Ridge", "Lasso", "ElasticNet",
                    "RandomForestRegressor", "GradientBoostingRegressor", "SVR"
                ])
                submitted = st.form_submit_button("Train Regression Model", use_container_width=True)
            if submitted:
                try:
                    result = ctrl.train_regression(reg_model)
                    if isinstance(result, tuple):
                        msg, data = result
                    else:
                        data = result
                        msg = data.get("message", "")
                    st.session_state["res_t3"] = (msg, data)
                    st.session_state["ml_model_name"] = reg_model
                    st.session_state["ml_task"] = "regression"
                    st.session_state["ml_model_trained"] = True
                except Exception as e:
                    st.session_state["res_t3"] = (f"Error: {e}", {})
        else:
            st.warning("Please complete preprocessing first.")
    with col_out:
        if "res_t3" in st.session_state:
            msg, data = st.session_state["res_t3"]
            if msg:
                st.code(msg, language=None)
            if data and "y_test" in data and "y_pred" in data:
                y_true = np.asarray(data["y_test"])
                y_pred = np.asarray(data["y_pred"])
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(y_true, y_pred, alpha=0.6, color="steelblue", edgecolors="white", s=40)
                lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
                ax.plot(lims, lims, "r--", linewidth=1.5, alpha=0.8, label="Perfect fit")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title(f"Regression – Actual vs Predicted  (R²={data.get('r2_test', 0):.4f})")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 4 - Classification
with tab4:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        if ctrl.is_preprocessed():
            with st.form("form_t4"):
                st.subheader("Classification")
                clf_model = st.selectbox("Model", [
                    "LogisticRegression", "RandomForestClassifier", "SVC", "KNeighborsClassifier"
                ])
                submitted = st.form_submit_button("Train Classification Model", use_container_width=True)
            if submitted:
                try:
                    result = ctrl.train_classification(clf_model)
                    if isinstance(result, tuple):
                        msg, data = result
                    else:
                        data = result
                        msg = data.get("message", "")
                    st.session_state["res_t4"] = (msg, data)
                    st.session_state["ml_model_name"] = clf_model
                    st.session_state["ml_task"] = "classification"
                    st.session_state["ml_model_trained"] = True
                except Exception as e:
                    st.session_state["res_t4"] = (f"Error: {e}", {})
        else:
            st.warning("Please complete preprocessing first.")
    with col_out:
        if "res_t4" in st.session_state:
            msg, data = st.session_state["res_t4"]
            if msg:
                st.code(msg, language=None)
            if data and "confusion_matrix" in data and data["confusion_matrix"] is not None:
                cm = np.asarray(data["confusion_matrix"])
                classes = data.get("classes", list(range(cm.shape[0])))
                fig, ax = plt.subplots(figsize=(max(5, len(classes)), max(4, len(classes))))
                im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
                plt.colorbar(im, ax=ax)
                ax.set_xticks(np.arange(len(classes)))
                ax.set_yticks(np.arange(len(classes)))
                ax.set_xticklabels([str(c) for c in classes], rotation=45, ha="right")
                ax.set_yticklabels([str(c) for c in classes])
                thresh = cm.max() / 2.0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black", fontsize=10)
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")
                ax.set_title(f"Confusion Matrix – {data.get('model_name', '')}  (acc={data.get('accuracy', 0):.3f})")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 5 - Clustering
with tab5:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        if ctrl.is_data_loaded():
            with st.form("form_t5"):
                st.subheader("Clustering")
                try:
                    all_cols = ctrl.get_columns()
                except Exception:
                    all_cols = []
                cluster_features = st.multiselect("Feature columns for clustering", all_cols,
                                                  default=all_cols[:2] if len(all_cols) >= 2 else all_cols)
                algorithm = st.selectbox("Algorithm", ["KMeans", "DBSCAN"])
                n_clusters = st.number_input("Number of clusters (KMeans)", value=3, min_value=2, max_value=20, step=1)
                eps = st.number_input("DBSCAN eps", value=0.5, step=0.05)
                min_samples = st.number_input("DBSCAN min_samples", value=5, min_value=1, step=1)
                submitted = st.form_submit_button("Run Clustering", use_container_width=True)
            if submitted:
                try:
                    result = ctrl.run_clustering(algorithm, int(n_clusters), cluster_features, eps, int(min_samples))
                    if isinstance(result, tuple):
                        msg, data = result
                    else:
                        data = result
                        msg = data.get("message", "")
                    st.session_state["res_t5"] = (msg, data)
                except Exception as e:
                    st.session_state["res_t5"] = (f"Error: {e}", {})
        else:
            st.warning("Please load data first.")
    with col_out:
        if "res_t5" in st.session_state:
            msg, data = st.session_state["res_t5"]
            if msg:
                st.code(msg, language=None)
            if data and "x" in data and "y" in data:
                fig, ax = plt.subplots(figsize=(8, 4))
                labels = data.get("labels", None)
                scatter = ax.scatter(data["x"], data["y"], c=labels, cmap="tab10", alpha=0.7, edgecolors="w", linewidth=0.5)
                if labels is not None:
                    plt.colorbar(scatter, ax=ax, label="Cluster")
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2")
                ax.set_title("Clustering Result")
                ax.grid(True, alpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 6 - PCA
with tab6:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        if ctrl.is_data_loaded():
            with st.form("form_t6"):
                st.subheader("Principal Component Analysis")
                try:
                    all_cols = ctrl.get_columns()
                except Exception:
                    all_cols = []
                pca_features = st.multiselect("Feature columns for PCA", all_cols,
                                              default=all_cols)
                n_components = st.slider("Number of components", 2, min(10, len(all_cols)) if all_cols else 2, 2)
                submitted = st.form_submit_button("Run PCA", use_container_width=True)
            if submitted:
                try:
                    result = ctrl.run_pca(n_components, pca_features)
                    if isinstance(result, tuple):
                        msg, data = result
                    else:
                        data = result
                        msg = data.get("message", "")
                    st.session_state["res_t6"] = (msg, data)
                except Exception as e:
                    st.session_state["res_t6"] = (f"Error: {e}", {})
        else:
            st.warning("Please load data first.")
    with col_out:
        if "res_t6" in st.session_state:
            msg, data = st.session_state["res_t6"]
            if msg:
                st.code(msg, language=None)
            if data and "explained_variance_ratio" in data:
                evr = data["explained_variance_ratio"]
                fig, ax = plt.subplots(figsize=(8, 4))
                x_pos = np.arange(1, len(evr) + 1)
                ax.bar(x_pos, [v * 100 for v in evr], color="steelblue", alpha=0.8)
                ax.set_xlabel("Principal Component")
                ax.set_ylabel("Explained Variance (%)")
                ax.set_title("PCA – Explained Variance Ratio")
                ax.set_xticks(x_pos)
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 7 - Cross-Validation
with tab7:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        if ctrl.is_model_trained():
            with st.form("form_t7"):
                st.subheader("Cross-Validation")
                model_name_cv = st.session_state.get("ml_model_name", "")
                task_cv = st.session_state.get("ml_task", "regression")
                st.text(f"Model: {model_name_cv}")
                st.text(f"Task: {task_cv}")
                cv_folds = st.number_input("Number of folds", value=5, min_value=2, max_value=20, step=1)
                submitted = st.form_submit_button("Run Cross-Validation", use_container_width=True)
            if submitted:
                try:
                    result = ctrl.run_cross_validation(model_name_cv, task_cv, int(cv_folds))
                    if isinstance(result, tuple):
                        msg, data = result
                    else:
                        data = result
                        msg = data.get("message", "")
                    st.session_state["res_t7"] = (msg, data)
                except Exception as e:
                    st.session_state["res_t7"] = (f"Error: {e}", {})
        else:
            st.warning("Please train a model first (Regression or Classification tab).")
    with col_out:
        if "res_t7" in st.session_state:
            msg, data = st.session_state["res_t7"]
            if msg:
                st.code(msg, language=None)
            if data and "scores" in data and len(data["scores"]) > 0:
                scores = np.asarray(data["scores"])
                fold_labels = [f"Fold {i+1}" for i in range(len(scores))]
                mean_val = float(np.mean(scores))
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(fold_labels, scores, color="steelblue", alpha=0.85, edgecolor="white")
                ax.axhline(mean_val, color="red", linestyle="--", linewidth=1.5,
                           label=f"Mean = {mean_val:.4f}")
                for bar, val in zip(bars, scores):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
                ax.set_ylabel(data.get("scoring", "Score"))
                ax.set_title(f"Cross-Validation Fold Scores – {data.get('model_name', '')}")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# Tab 8 - Predict
with tab8:
    col_in, col_out = st.columns([1, 2])
    with col_in:
        if ctrl.is_model_trained():
            st.subheader("Make Predictions")
            features = st.session_state.get("ml_features", [])
            with st.form("form_t8"):
                input_values = []
                for feat in features:
                    val = st.number_input(f"{feat}", value=0.0, step=0.1, key=f"pred_{feat}")
                    input_values.append(val)
                submitted = st.form_submit_button("Predict", use_container_width=True)
            if submitted:
                try:
                    result = ctrl.predict(input_values)
                    if isinstance(result, tuple):
                        msg, data = result
                    else:
                        data = result
                        msg = data.get("message", "")
                    st.session_state["res_t8"] = (msg, data)
                except Exception as e:
                    st.session_state["res_t8"] = (f"Error: {e}", {})
        else:
            st.warning("Please train a model first.")
    with col_out:
        if "res_t8" in st.session_state:
            msg, data = st.session_state["res_t8"]
            if msg:
                st.code(msg, language=None)
