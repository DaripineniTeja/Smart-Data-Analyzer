import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import zscore

def ml_model_prediction_tab(df):
    st.subheader("🤖 Machine Learning - Predictive Modeling")

    target = st.selectbox("🎯 Select Target Variable", df.columns)
    features = st.multiselect("📥 Select Feature Columns", [col for col in df.columns if col != target])

    if target and features:
        model_type = st.radio("📚 Problem Type", ["Classification", "Regression"])

        X = df[features]
        y = df[target]

        # Handle categorical data
        X = pd.get_dummies(X, drop_first=True)
        if y.dtype == 'O':
            y = pd.factorize(y)[0]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier() if model_type == "Classification" else RandomForestRegressor()

        if st.button("🚀 Train Model"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success("✅ Model Trained Successfully!")

            # Classification Metrics
            if model_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                st.markdown("### 📈 Classification Metrics")
                st.write(f"**Accuracy:** {acc:.4f}")
                st.write(f"**Precision:** {prec:.4f}")
                st.write(f"**Recall:** {rec:.4f}")
                st.write(f"**F1 Score:** {f1:.4f}")

                st.markdown("### 🔲 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

                st.markdown("### 📋 Classification Report")
                st.text(classification_report(y_test, y_pred))

            # Regression Metrics
            else:
                r2 = r2_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mae = mean_absolute_error(y_test, y_pred)

                st.markdown("### 📉 Regression Metrics")
                st.write(f"**R² Score:** {r2:.4f}")
                st.write(f"**RMSE:** {rmse:.4f}")
                st.write(f"**MAE:** {mae:.4f}")

                st.markdown("### 📊 Actual vs Predicted Plot")
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, alpha=0.6)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                st.pyplot(fig)

                # Optional: Show Z-scores
                st.markdown("### ⚠️ Z-Score for Outlier Detection")
                z_scores = np.abs(zscore(X))
                outliers = (z_scores > 3).sum(axis=1)
                st.write(f"Outlier Rows (Z-score > 3): {np.sum(outliers > 0)} out of {X.shape[0]}")

            # Save the trained model
            joblib.dump(model, "trained_model.pkl")
            with open("trained_model.pkl", "rb") as f:
                st.download_button("📥 Download Trained Model", f, file_name="model.pkl")
