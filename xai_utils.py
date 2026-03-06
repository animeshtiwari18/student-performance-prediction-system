import shap
import pandas as pd

def get_shap_explanation(model, input_data, feature_names):
    """
    Returns:
    - raw shap values
    - sorted importance
    - positive & negative contributions
    """

    explainer = shap.TreeExplainer(model)

    X = pd.DataFrame([input_data])

    shap_values = explainer.shap_values(X)

    # For classification → shap_values is list per class
    # We take the predicted class (highest probability)
    predicted_class_index = model.predict_proba(X)[0].argmax()

    values = shap_values[predicted_class_index][0]

    explanation = dict(zip(feature_names, values))

    # Sort by absolute impact
    sorted_explanation = dict(
        sorted(explanation.items(),
               key=lambda x: abs(x[1]),
               reverse=True)
    )

    positive = {k: v for k, v in sorted_explanation.items() if v > 0}
    negative = {k: v for k, v in sorted_explanation.items() if v < 0}

    return {
        "all_features": explanation,
        "sorted_by_impact": sorted_explanation,
        "positive_impact": positive,
        "negative_impact": negative
    }