from flask import Flask, request, render_template, send_file, jsonify
import os
import pandas as pd
import joblib
import logging
import json
import traceback
from werkzeug.utils import secure_filename
from train import prepare_data, train_rf_model as train_model
from test import load_models, predict_matches, separate_results
from merging import download_merged_names as run_merging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define folder paths
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
SEPARATED_FOLDER = 'seperated'
CLEANED_FOLDER = 'cleaned'
MODELS_FOLDER = 'models'

FOLDERS_TO_CLEAR = [UPLOAD_FOLDER, SEPARATED_FOLDER, RESULT_FOLDER, CLEANED_FOLDER]

# Create folders if they don't exist
for folder in [UPLOAD_FOLDER, RESULT_FOLDER, SEPARATED_FOLDER, CLEANED_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)


def get_available_downloads(name_without_ext):
    downloads = {}
    result_files = {
        'complete_results': (RESULT_FOLDER, f'{name_without_ext}_new_data_output.csv'),
        'matched_names': (SEPARATED_FOLDER, f'{name_without_ext}_predicted_1.csv'),
        'unmatched_names': (SEPARATED_FOLDER, f'{name_without_ext}_predicted_0.csv'),
        'merged_names': (CLEANED_FOLDER, f'{name_without_ext}_cleaned_predicted_1.csv')
    }
    for key, (folder, filename) in result_files.items():
        if os.path.exists(os.path.join(folder, filename)):
            downloads[key] = f'/download_{key}?name_without_ext={name_without_ext}'
    return downloads


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train', methods=['POST'])
def train_model_route():
    try:
        if 'train_file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['train_file']
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        logger.info(f"Training file saved to: {filepath}")

        df = pd.read_csv(filepath)

        required_columns = ['name1', 'name2', 'is_match']
        if any(col not in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")

        X, y, tfidf, sbert, feature_cols = prepare_data(df)

        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        model = train_model(X_train, y_train)

        model_path = os.path.join(MODELS_FOLDER, 'xgb_rf_model.json')
        model.save_model(model_path)

        joblib.dump(tfidf, os.path.join(MODELS_FOLDER, 'tfidf_vectorizer.pkl'))
        joblib.dump(sbert, os.path.join(MODELS_FOLDER, 'sbert_model.pkl'))
        joblib.dump(feature_cols, os.path.join(MODELS_FOLDER, 'features_used.pkl'))

        return jsonify({"message": "Model trained successfully!", "status": "success"}), 200

    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route('/test', methods=['POST'])
def test_model_route():
    try:
        if 'test_file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['test_file']
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        name_without_ext, ext = os.path.splitext(filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        ext = ext.lower()

        if ext == '.csv':
            df = pd.read_csv(filepath)

            required_columns = ['name1', 'name2']
            if any(col not in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")

            return process_dataframe(df, name_without_ext)

        elif ext in ['.xls', '.xlsx']:
            try:
                excel_file = pd.ExcelFile(filepath)
                sheet_names = excel_file.sheet_names
                return jsonify({
                    "status": "excel",
                    "sheets": sheet_names,
                    "filename": filename
                }), 200
            except Exception as e:
                logger.error(f"Error reading Excel file: {e}")
                return jsonify({"error": f"Failed to read Excel file: {str(e)}"}), 500

        else:
            return jsonify({"error": "Unsupported file type. Please upload a CSV or Excel file."}), 400

    except Exception as e:
        logger.error(f"Testing error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route('/process_excel', methods=['POST'])
def process_excel_sheet():
    try:
        filename = request.form.get('filename')
        sheet_name = request.form.get('sheet_name')

        if not filename or not sheet_name:
            return jsonify({"error": "Missing filename or sheet_name"}), 400

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "Uploaded file not found"}), 400

        df = pd.read_excel(filepath, sheet_name=sheet_name)

        required_columns = ['name1', 'name2']
        if any(col not in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")

        name_without_ext, _ = os.path.splitext(filename)
        return process_dataframe(df, name_without_ext)

    except Exception as e:
        logger.error(f"Excel sheet processing error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


def process_dataframe(df, name_without_ext):
    try:
        model, tfidf, sbert, features = load_models()
        results_df = predict_matches(df, model, tfidf, sbert, features)

        result_path = os.path.join(RESULT_FOLDER, f'{name_without_ext}_new_data_output.csv')
        results_df.to_csv(result_path, index=False)

        matched, unmatched, message = separate_results(results_df)
        matched_path = os.path.join(SEPARATED_FOLDER, f'{name_without_ext}_predicted_1.csv')
        unmatched_path = os.path.join(SEPARATED_FOLDER, f'{name_without_ext}_predicted_0.csv')
        matched.to_csv(matched_path, index=False)
        unmatched.to_csv(unmatched_path, index=False)

        merging_result = {"status": "skipped", "message": "Merging not executed."}
        try:
         merging_result = run_merging(name_without_ext)
        except Exception as error_message:
         logger.warning(f"Merging step failed: {error_message}")
         merging_result = {
         "status": "error",
         "error_message": f'Cannot find the column - {str(error_message)}',
        "traceback": traceback.format_exc()
            }

# Standardize the final response for frontend
        if merging_result.get("status") == "error":
          merging_result = {
        "status": "error",
        "message": merging_result.get("error_message", "Merging failed")
    }


        return app.response_class(
            response=json.dumps({
                "message": f"Done! {message}",
                "status": "success",
                "downloads": get_available_downloads(name_without_ext),
                "merging": merging_result
            }, ensure_ascii=False),
            status=200,
            mimetype='application/json'
        )

    except Exception as e:
        logger.error(f"Data processing error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route('/clear_cache', methods=['GET', 'POST'])
def clear_files():
    try:
        for folder in FOLDERS_TO_CLEAR:
            for f in os.listdir(folder):
                path = os.path.join(folder, f)
                if os.path.isfile(path):
                    os.remove(path)
        return jsonify({"status": "success", "message": "Cache cleared"}), 200
    except Exception as e:
        logger.error(f"Cache clearing error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# Download routes using query param
@app.route('/download_complete_results')
def download_complete_results():
    return download_file_helper(RESULT_FOLDER, "_new_data_output.csv", "complete_results.csv")


@app.route('/download_matched_names')
def download_matched_names():
    return download_file_helper(SEPARATED_FOLDER, "_predicted_1.csv", "matched_names.csv")


@app.route('/download_unmatched_names')
def download_unmatched_names():
    return download_file_helper(SEPARATED_FOLDER, "_predicted_0.csv", "unmatched_names.csv")


@app.route('/download_merged_names')
def download_merged_names():
    return download_file_helper(CLEANED_FOLDER, "_cleaned_predicted_1.csv", "merged_names.csv")


def download_file_helper(folder, file_suffix, download_name):
    name = request.args.get("name_without_ext")
    if not name:
        return jsonify({"error": "Missing name parameter"}), 400
    path = os.path.join(folder, f"{name}{file_suffix}")
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name=f"{name}_{download_name}")
    return jsonify({"error": "File not found"}), 404


if __name__ == '__main__':
    app.run(debug=True)
