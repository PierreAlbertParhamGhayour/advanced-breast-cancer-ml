# src/explainability.py

import shap
import matplotlib.pyplot as plt

def explain_with_shap(model, X_train, X_test):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png")
