"# My Project" 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

st.set_page_config(
    page_title="Walmart Data Science Project",
    layout="wide"
)

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown("""
    <style>
    h1 {
        color: #0071ce;
        font-family: 'Trebuchet MS', sans-serif;
        font-size: 40px;
    }
    h2 {
        color: #ffc220;
        font-family: 'Verdana', sans-serif;
    }
    .stButton>button {
        background-color: #0071ce;
        color: white;
        border-radius: 8px;
        font-size: 16px;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# SESSION STATE INIT
# -----------------------------
defaults = {
    "logged_in": False,
    "df": None,
    "nav": "Upload Dataset",
    "active_subsection": None,
    "selected_features": None,
    "target_column": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "model_results": [],
    "task_type": "Regression"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# LOGIN PAGE
# -----------------------------
def login_page():
    st.image("walmart_logo.png", width=150)
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Login"):
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")
    with c2:
        if st.button("Forgot Password"):
            st.info("Please contact the administrator")

# -----------------------------
# MAIN APP
# -----------------------------
def main_app():
    st.image("walmart_logo.png", width=150)
    st.title("Walmart Data Science Project")
    st.markdown("<hr style='border:2px solid #0071ce'>", unsafe_allow_html=True)

    with st.sidebar:
        st.image("walmart_logo.png", width=120)
        st.header("Navigation")
        st.markdown("### Walmart Data Science Dashboard")
        st.markdown("Insights powered by Machine Learning 📊")

        st.session_state.task_type = st.radio("Select Task Type", ["Regression", "Classification"])

        if st.button("Upload Dataset"): st.session_state.nav = "Upload Dataset"
        if st.button("Data Preprocessing"): st.session_state.nav = "Data Preprocessing"
        if st.button("Feature Selection"): st.session_state.nav = "Feature Selection"
        if st.button("Train/Test Split"): st.session_state.nav = "Train/Test Split"
        if st.button("Model Selection"): st.session_state.nav = "Model Selection"
        if st.button("Model Comparison"): st.session_state.nav = "Model Comparison"
        if st.button("Final Output"): st.session_state.nav = "Final Output"

    nav = st.session_state.nav

    # =========================
    # UPLOAD DATASET
    # =========================
    if nav == "Upload Dataset":
        st.header("Dataset Upload")
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Clear Dataset"):
                for key in ["df", "selected_features", "target_column", "active_subsection"]:
                    st.session_state[key] = None
                st.success("Dataset cleared")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success("Dataset uploaded successfully")

        if st.session_state.df is not None:
            df = st.session_state.df
            st.subheader("Dataset Overview")
            b1, b2, b3, b4 = st.columns(4)
            if b1.button("First 5 Rows"): st.session_state.active_subsection = "head"
            if b2.button("Features"): st.session_state.active_subsection = "columns"
            if b3.button("Statistical Summary"): st.session_state.active_subsection = "summary"
            if b4.button("Dimensions"): st.session_state.active_subsection = "shape"

            if st.session_state.active_subsection == "head":
                st.dataframe(df.head(), width="stretch")
            elif st.session_state.active_subsection == "columns":
                st.dataframe(pd.DataFrame(df.columns, columns=["Feature Name"]), width="stretch")
            elif st.session_state.active_subsection == "summary":
                st.dataframe(df.describe(), width="stretch")
            elif st.session_state.active_subsection == "shape":
                rows, cols = df.shape
                st.info(f"Number of Rows: {rows}\nNumber of Columns: {cols}\nDimensions: ({rows}, {cols})")

    # =========================
    # DATA PREPROCESSING
    # =========================
    elif nav == "Data Preprocessing":
        if st.session_state.df is None:
            st.warning("Please upload a dataset first.")
            st.stop()
        df = st.session_state.df
        st.header("Data Preprocessing")
        p1, p2, p3 = st.columns(3)
        if p1.button("Missing Data"): st.session_state.active_subsection = "missing"
        if p2.button("Outliers"): st.session_state.active_subsection = "outliers"
        if p3.button("Normalization"): st.session_state.active_subsection = "normalization"

       # =========================
        # Missing Data Dashboard
        # =========================
        if st.session_state.active_subsection == "missing":
            st.subheader("Missing Data Techniques")

            # Ensure df is defined
            df = st.session_state.df
            df_missing = df.copy()

            # Define numeric columns once, outside the button block
            num_cols = df.select_dtypes(include="number").columns

            # Add a default "None" option so user must choose explicitly
            method = st.radio(
                "Choose a technique:",
                ["None (Select a technique)", "Mean Imputation", "Median Imputation", "Mode Imputation", "Drop Missing Rows"]
            )

            if st.button("Apply Missing Data Technique"):
                st.session_state["missing_applied"] = True

                if method == "None (Select a technique)":
                    st.warning("⚠️ Please select a valid technique to apply.")

                elif method == "Mean Imputation":
                    df_missing[num_cols] = df_missing[num_cols].fillna(df_missing[num_cols].mean())
                    st.dataframe(df_missing.describe(), width="stretch")

                elif method == "Median Imputation":
                    df_missing[num_cols] = df_missing[num_cols].fillna(df_missing[num_cols].median())
                    st.dataframe(df_missing.describe(), width="stretch")

                elif method == "Mode Imputation":
                    for col in df_missing.columns:
                        mode_val = df_missing[col].mode()
                        if not mode_val.empty:
                            df_missing[col] = df_missing[col].fillna(mode_val[0])
                    st.dataframe(df_missing.describe(include="all"), width="stretch")

                elif method == "Drop Missing Rows":
                    df_dropped = df_missing.dropna()
                    st.dataframe(df_dropped.describe(), width="stretch")

        # Outliers Dashboard
        elif st.session_state.active_subsection == "outliers":
            st.subheader("Outlier Detection Techniques")
                #Add a default "None" option so user must choose explicitly
            method = st.radio(
                "Choose a technique:",
                ["None (Select a technique)", "Z-Score Method", "IQR Method", "Percentile Clipping", "Isolation (NaN Flag)"]
            )
            if st.button("Apply Outlier Detection"):
                df_out = df.copy()
                num_cols = df.select_dtypes(include="number").columns
                if method == "Z-Score Method":
                    z_scores = np.abs((df_out[num_cols] - df_out[num_cols].mean()) / df_out[num_cols].std())
                    outliers = (z_scores > 3).sum()
                    st.dataframe(pd.DataFrame(outliers, columns=["Outlier Count"]), width="stretch")
                elif method == "IQR Method":
                    Q1 = df_out[num_cols].quantile(0.25)
                    Q3 = df_out[num_cols].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df_out[num_cols] < (Q1 - 1.5 * IQR)) | (df_out[num_cols] > (Q3 + 1.5 * IQR))).sum()
                    st.dataframe(pd.DataFrame(outliers, columns=["Outlier Count"]), width="stretch")

                    # 📊 Enhanced Boxplot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.boxplot(
                        [df_out[col].dropna() for col in num_cols],
                        patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='blue'),
                        whiskerprops=dict(color='blue'),
                        capprops=dict(color='blue'),
                        flierprops=dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none'),
                        medianprops=dict(color='darkblue'),
                        tick_labels=num_cols
                    )
                    ax.set_title("IQR Outlier Detection – Enhanced Boxplot", fontsize=14, color="#0071ce")
                    ax.set_ylabel("Value Range")
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig)

                elif method == "Percentile Clipping":
                    lower = df_out[num_cols].quantile(0.01)
                    upper = df_out[num_cols].quantile(0.99)
                    df_clipped = df_out[num_cols].clip(lower=lower, upper=upper, axis=1)
                    st.dataframe(df_clipped.describe(), width="stretch")

                elif method == "Isolation (NaN Flag)":
                    Q1 = df_out[num_cols].quantile(0.25)
                    Q3 = df_out[num_cols].quantile(0.75)
                    IQR = Q3 - Q1
                    mask = (df_out[num_cols] < (Q1 - 1.5 * IQR)) | (df_out[num_cols] > (Q3 + 1.5 * IQR))
                    df_flagged = df_out.copy()
                    df_flagged[num_cols] = df_flagged[num_cols].mask(mask)
                    st.dataframe(df_flagged.describe(), width="stretch")

        # Normalization Dashboard
        elif st.session_state.active_subsection == "normalization":
            st.subheader("Normalization Techniques")
            # Add a default "None" option so user must choose explicitly
            method = st.radio(
                "Choose a technique:",
                ["None (Select a technique)", "Min-Max Scaling (0 to 1)", "Min-Max Scaling (-1 to 1)"]
            )
            if st.button("Apply Normalization"):
                df_norm = df.copy()
                num_cols = df.select_dtypes(include="number").columns

                if method == "Min-Max Scaling (0 to 1)":
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    df_norm[num_cols] = scaler.fit_transform(df[num_cols])
        
                elif method == "Min-Max Scaling (-1 to 1)":
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    df_norm[num_cols] = scaler.fit_transform(df[num_cols])
            
                st.session_state.df = df_norm 
                st.success("Normalization applied! Dataset updated for training/testing.") 
                st.dataframe(df_norm.describe(), width="stretch")

    # =========================
    # FEATURE SELECTION
    # =========================
    elif nav == "Feature Selection":
        if st.session_state.df is None:
            st.warning("Please upload a dataset first.")
            st.stop()
        df = st.session_state.df
        st.header("Feature Selection")
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_columns) < 2:
            st.error("Not enough numeric columns for feature selection.")
            st.stop()
        target = st.selectbox("Select Target Column (Numeric Only)", numeric_columns)
        st.session_state.target_column = target

        if st.session_state.task_type == "Classification": 
            df['Target_Class'] = pd.qcut(df[target], q=3, labels=["Low", "Medium", "High"]) 
            st.session_state.target_column = "Target_Class"

        method = st.selectbox("Select Feature Selection Technique", ["Correlation", "Variance Threshold"])
        features = df.drop(columns=[target]).select_dtypes(include="number")
        y = df[target]

        if method == "Correlation":
            corr = features.corrwith(y).abs().sort_values(ascending=False)
            top_n = st.slider("Number of Features", 1, len(corr), min(5, len(corr)))
            selected = corr.head(top_n)

            st.subheader("Selected Features (Correlation)")
            result_df = pd.DataFrame({
                "Dependent Variable": [target] * len(selected),
                "Independent Feature": selected.index,
                "Correlation": selected.values
            })
            st.dataframe(result_df, width="stretch")
            st.session_state.selected_features = list(selected.index)

        elif method == "Variance Threshold":
            threshold = st.slider("Variance Threshold", 0.0, 0.2, 0.01)
            selector = VarianceThreshold(threshold)
            selector.fit(features)
            selected_cols = features.columns[selector.get_support()]
            st.subheader("Selected Features (Variance Threshold)")
            st.dataframe(pd.DataFrame(selected_cols, columns=["Feature"]), width="stretch")
            st.session_state.selected_features = list(selected_cols)

    # =========================
    # TRAIN/TEST SPLIT
    # =========================
    elif nav == "Train/Test Split":
        if st.session_state.df is None or st.session_state.selected_features is None:
            st.warning("Please complete feature selection first.")
            st.stop()

        df = st.session_state.df
        clean_df = df.dropna(subset=st.session_state.selected_features + [st.session_state.target_column])
        X = clean_df[st.session_state.selected_features]
        y = clean_df[st.session_state.target_column]

        st.header("Train/Test Split")
        test_size = st.slider("Test Set Percentage", 10, 50, 20)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)
        st.session_state.X_train, st.session_state.X_test = X_train, X_test
        st.session_state.y_train, st.session_state.y_test = y_train, y_test

        st.success("Train/Test split completed!")

        st.subheader("Training Data Preview")
        train_preview = pd.concat([X_train, y_train.rename("Target")], axis=1)
        st.dataframe(train_preview.head(), width="stretch")

        st.subheader("Testing Data Preview")
        test_preview = pd.concat([X_test, y_test.rename("Target")], axis=1)
        st.dataframe(test_preview.head(), width="stretch")

    # =========================
    # MODEL SELECTION
    # =========================
    elif nav == "Model Selection":
        if st.session_state.X_train is None:
            st.warning("Please complete train/test split first.")
            st.stop()

        if st.session_state.task_type == "Regression":
            model_choice = st.selectbox("Choose Regression Model", ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"])
            if model_choice == "Linear Regression": model = LinearRegression()
            elif model_choice == "Decision Tree": model = DecisionTreeRegressor()
            elif model_choice == "Random Forest": model = RandomForestRegressor()
            elif model_choice == "XGBoost": model = XGBRegressor()
        else:
            model_choice = st.selectbox("Choose Classification Model", ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"])
            if model_choice == "Logistic Regression": model = LogisticRegression(max_iter=500)
            elif model_choice == "Decision Tree": model = DecisionTreeClassifier()
            elif model_choice == "Random Forest": model = RandomForestClassifier()
            elif model_choice == "XGBoost": model = XGBClassifier()

        model.fit(st.session_state.X_train, st.session_state.y_train)
        y_pred_train = model.predict(st.session_state.X_train)
        y_pred_test = model.predict(st.session_state.X_test)

        st.subheader("Model Evaluation")

        if st.session_state.task_type == "Regression":
            st.write(f"Train R² Score: {r2_score(st.session_state.y_train, y_pred_train):.2f}")
            st.write(f"Test R² Score: {r2_score(st.session_state.y_test, y_pred_test):.2f}")
            rmse = np.sqrt(mean_squared_error(st.session_state.y_test, y_pred_test))
            st.write(f"Test RMSE: {rmse:.2f}")

        else:  # Classification
            # Confusion Matrix
            cm = confusion_matrix(st.session_state.y_test, y_pred_test)
            st.write("Confusion Matrix:")
            st.dataframe(pd.DataFrame(cm,
                                  index=["Actual Negative", "Actual Positive"],
                                  columns=["Predicted Negative", "Predicted Positive"]),
                     width="stretch")

            # Metrics
            acc = accuracy_score(st.session_state.y_test, y_pred_test)
            prec = precision_score(st.session_state.y_test, y_pred_test, average="binary")
            rec = recall_score(st.session_state.y_test, y_pred_test, average="binary")
            f1 = (2 * prec * rec) / (prec + rec)

            st.write(f"Accuracy: {acc:.2f}")
            st.write(f"Precision: {prec:.2f}")
            st.write(f"Recall: {rec:.2f}")
            st.write(f"F1 Score: {f1:.2f}")

            # ROC Curve (only for binary classification)
            if len(set(st.session_state.y_test)) == 2:
                y_score = model.predict_proba(st.session_state.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(st.session_state.y_test, y_score)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color="#0071ce", lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend(loc="lower right")
                st.pyplot(fig)
            else:
               st.warning("ROC curve only available for binary classification.")




    # =========================  
    # MODEL COMPARISON 
    # ========================= 
    elif nav == "Model Comparison":
        if st.session_state.X_train is None:
            st.warning("Please complete train/test split first.")
            st.stop()

        st.header("Model Comparison")
        if st.session_state.task_type == "Regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBoost": XGBRegressor()
            }
        else:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=500),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier()
            }

        results = []
        for name, model in models.items():
            model.fit(st.session_state.X_train, st.session_state.y_train)
            y_pred = model.predict(st.session_state.X_test)

            if st.session_state.task_type == "Regression":
                r2 = r2_score(st.session_state.y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(st.session_state.y_test, y_pred))
                results.append({"Model": name, "R² Score": round(r2, 2), "RMSE": round(rmse, 2)})
            else:
                acc = accuracy_score(st.session_state.y_test, y_pred)
                prec = precision_score(st.session_state.y_test, y_pred, average="binary")
                rec = recall_score(st.session_state.y_test, y_pred, average="binary")
                f1 = (2 * prec * rec) / (prec + rec)

                if len(set(st.session_state.y_test)) == 2:
                    y_score = model.predict_proba(st.session_state.X_test)[:, 1]
                    auc_score = roc_auc_score(st.session_state.y_test, y_score)
                else:
                    auc_score = None

                results.append({
                    "Model": name,
                    "Accuracy": round(acc, 2),
                    "Precision": round(prec, 2),
                    "Recall": round(rec, 2),
                    "F1 Score": round(f1, 2),
                    "AUC": round(auc_score, 2) if auc_score is not None else "N/A"
                })

        st.session_state.model_results = results
        results_df = pd.DataFrame(results)

        st.subheader("Model Performance Comparison")
        st.dataframe(results_df, width="stretch")

        # Visualization (Classification Metrics Only)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Pick available classification metric columns
        metric_candidates = ["Accuracy", "F1 Score", "Precision", "Recall", "AUC"]
        available_metrics = [m for m in metric_candidates if m in results_df.columns]

        if available_metrics:
            results_df.plot(
                x="Model",
                y=available_metrics,
                kind="bar",
                ax=ax,
                color=["#0071ce", "#ffc220", "#28a745", "#d63384", "#6f42c1"][:len(available_metrics)]
            )
            ax.set_title("Model Comparison - Classification Metrics")
            ax.set_ylabel("Score")
            ax.legend(loc="best")
        else:
            st.warning("No classification metrics available to plot.")
            ax.set_xticklabels(results_df["Model"], rotation=45, ha="right")

        st.pyplot(fig)


    # =========================
    # FINAL OUTPUT
    # =========================
    elif nav == "Final Output":
        if not st.session_state.model_results:
            st.warning("Please run model comparison first.")
            st.stop()

        st.header("Best Model Summary")
        if st.session_state.task_type == "Regression":
            best_model = max(st.session_state.model_results, key=lambda x: x["R² Score"])
            st.success(f"🏆 Best Model: {best_model['Model']}")
            st.write(f"R² Score: {best_model['R² Score']}")
            st.write(f"RMSE: {best_model['RMSE']}")
        else:
            best_model = max(st.session_state.model_results, key=lambda x: x["Accuracy"])
            st.success(f"🏆 Best Model: {best_model['Model']}")
            st.write(f"Accuracy: {best_model['Accuracy']}")

        summary_df = pd.DataFrame([best_model])
        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Best Model Summary", csv, "best_model.csv", "text/csv")

# -----------------------------
# FLOW CONTROL
# -----------------------------
if st.session_state.logged_in:
    main_app()
else:
    login_page()
