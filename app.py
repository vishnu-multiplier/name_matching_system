from flask import Flask, request, render_template, send_file, jsonify
import os
import pandas as pd
import joblib
import logging
import json
from werkzeug.utils import secure_filename
from train import prepare_data, train_model
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

# Create folders
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
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if file.filename is None:
            return jsonify({"error": "Invalid filename"}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        logger.info(f"Training file saved to: {filepath}")

        df = pd.read_csv(filepath)
        required_columns = ['name1', 'name2', 'is_match']
        if any(col not in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")

        X_train, y_train, tfidf, sbert = prepare_data(df)
        model = train_model(X_train, y_train, tfidf, sbert)

        joblib.dump(model, os.path.join(MODELS_FOLDER, 'xgb_model.pkl'))
        joblib.dump(tfidf, os.path.join(MODELS_FOLDER, 'tfidf_vectorizer.pkl'))
        joblib.dump(sbert, os.path.join(MODELS_FOLDER, 'sbert_model.pkl'))
        joblib.dump(X_train.columns.tolist(), os.path.join(MODELS_FOLDER, 'features_used.pkl'))

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
        name_without_ext, _ = os.path.splitext(filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        required_columns = ['name1', 'name2']
        if any(col not in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")

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
        except Exception as merge_error:
            logger.warning(f"Merging step failed: {merge_error}")
            merging_result = {
                "status": "error",
                "message": str(merge_error)
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
        logger.error(f"Testing error: {e}")
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
        return jsonify({"status": "error", "message": str(e)}), 500

# Download routes using query param
@app.route('/download_complete_results')
def download_complete_results():
    name = request.args.get("name_without_ext")
    path = os.path.join(RESULT_FOLDER, f"{name}_new_data_output.csv")
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name=f"{name}_complete_results.csv")
    return jsonify({"error": "File not found"}), 404

@app.route('/download_matched_names')
def download_matched_names():
    name = request.args.get("name_without_ext")
    path = os.path.join(SEPARATED_FOLDER, f"{name}_predicted_1.csv")
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name=f"{name}_matched_names.csv")
    return jsonify({"error": "File not found"}), 404

@app.route('/download_unmatched_names')
def download_unmatched_names():
    name = request.args.get("name_without_ext")
    path = os.path.join(SEPARATED_FOLDER, f"{name}_predicted_0.csv")
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name=f"{name}_unmatched_names.csv")
    return jsonify({"error": "File not found"}), 404

@app.route('/download_merged_names')
def download_merged_names():
    name = request.args.get("name_without_ext")
    path = os.path.join(CLEANED_FOLDER, f"{name}_cleaned_predicted_1.csv")
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name=f"{name}_merged_names.csv")
    return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
