import joblib
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import logging
import os

# ----------- Logging Setup -----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------- Paths -----------
MODEL_PATH = "models/xgb_model.pkl"
FEATURES_PATH = "models/features_used.pkl"
OUTPUT_IMAGE = "features/feature_importance_all.png"
OUTPUT_EXCEL = "features/feature_importance_all.xlsx"

# ----------- Load Model and Features -----------
def load_model_and_features():
    logger.info("Loading model and features...")
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    logger.info("Loaded model and features successfully.")
    return model, features

# ----------- Plot and Save Feature Importances -----------
def plot_all_importances(model, feature_names):
    booster = model.get_booster()
    importance_types = ['gain', 'weight', 'cover']
    importance_data = {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('XGBoost Feature Importances by Type', fontsize=16)

    for i, imp_type in enumerate(importance_types):
        logger.info(f"Generating importance by: {imp_type}")
        imp_dict = booster.get_score(importance_type=imp_type)
        imp_series = pd.Series(imp_dict)

        # Align with expected feature order
        aligned = imp_series.reindex(feature_names).fillna(0).sort_values(ascending=True)
        importance_data[imp_type] = aligned.sort_values(ascending=False)

        # Plot
        aligned.plot(kind='barh', ax=axes[i])
        axes[i].set_title(f'{imp_type.capitalize()} Importance')
        axes[i].set_xlabel('Importance')
        axes[i].grid(True)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(OUTPUT_IMAGE)
    plt.show()
    logger.info(f"Combined feature importance plot saved to: {OUTPUT_IMAGE}")

    # Save all importance values to one Excel file (multiple sheets)
    with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
        for imp_type, df in importance_data.items():
            df.to_frame(name=f"{imp_type}_importance").to_excel(writer, sheet_name=imp_type)

    logger.info(f"Feature importance Excel saved to: {OUTPUT_EXCEL}")

# ----------- Main -----------
def main():
    try:
        model, features = load_model_and_features()
        plot_all_importances(model, features)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
