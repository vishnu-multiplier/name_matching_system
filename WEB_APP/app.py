from flask import Flask, request, render_template, send_file, jsonify
import os
import pandas as pd
import joblib
import logging
from werkzeug.utils import secure_filename
from train import prepare_data, train_model
from test import load_models, predict_matches, separate_results

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

# Create required directories
for folder in [UPLOAD_FOLDER, RESULT_FOLDER, SEPARATED_FOLDER, CLEANED_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def get_available_downloads():
    """Get list of available result files for download."""
    downloads = {}
    result_files = {
        'complete_results': (RESULT_FOLDER, 'new_data_output.csv'),
        'matched_names': (SEPARATED_FOLDER, 'predicted_1.csv'),
        'unmatched_names': (SEPARATED_FOLDER, 'predicted_0.csv'),
        'merged_names': (CLEANED_FOLDER, 'cleaned_predicted_1.csv')
    }
    
    for key, (folder, filename) in result_files.items():
        if os.path.exists(os.path.join(folder, filename)):
            downloads[key] = f'/download_{key}'
    
    return downloads

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model_route():
    """Handle model training requests."""
    try:
        # Validate file upload
        if 'train_file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['train_file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        logger.info(f"Training file saved to: {filepath}")

        try:
            # Load and validate data
            logger.info("Loading training data...")
            df = pd.read_csv(filepath)
            
            # Verify required columns
            required_columns = ['name1', 'name2', 'is_match']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Prepare data
            logger.info("Preparing data for training...")
            X_train, y_train, tfidf, sbert = prepare_data(df)
            
            # Train model
            logger.info("Training model...")
            model = train_model(X_train, y_train, tfidf, sbert)
            
            # Save models
            logger.info("Saving models...")
            joblib.dump(model, os.path.join(MODELS_FOLDER, 'xgb_model.pkl'))
            joblib.dump(tfidf, os.path.join(MODELS_FOLDER, 'tfidf_vectorizer.pkl'))
            joblib.dump(sbert, os.path.join(MODELS_FOLDER, 'sbert_model.pkl'))
            joblib.dump(X_train.columns.tolist(), os.path.join(MODELS_FOLDER, 'features_used.pkl'))
            
            return jsonify({
                "message": "Model trained successfully!",
                "status": "success"
            }), 200

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500

    except Exception as e:
        logger.error(f"Error processing training request: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/test', methods=['POST'])
def test_model_route():
    """Handle model testing requests."""
    try:
        # Validate file upload
        if 'test_file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['test_file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        logger.info(f"Test file saved to: {filepath}")

        try:
            # Load test data
            logger.info("Loading test data...")
            df = pd.read_csv(filepath)
            
            # Verify required columns
            required_columns = ['name1', 'name2']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Load models
            logger.info("Loading models...")
            model, tfidf, sbert, features = load_models()
            
            # Make predictions
            logger.info("Making predictions...")
            results_df = predict_matches(df, model, tfidf, sbert, features)
            
            # Save complete results
            logger.info("Saving results...")
            results_path = os.path.join(RESULT_FOLDER, 'new_data_output.csv')
            results_df.to_csv(results_path, index=False)
            
            # Separate and save matched/unmatched results
            matched, unmatched = separate_results(results_df)
            matched.to_csv(os.path.join(SEPARATED_FOLDER, 'predicted_1.csv'), index=False)
            unmatched.to_csv(os.path.join(SEPARATED_FOLDER, 'predicted_0.csv'), index=False)
            
            return jsonify({
                "message": "Testing completed successfully!",
                "status": "success",
                "downloads": get_available_downloads()
            }), 200

        except Exception as e:
            logger.error(f"Error during testing: {str(e)}")
            return jsonify({
                "error": str(e),
                "status": "error"
            }), 500

    except Exception as e:
        logger.error(f"Error processing test request: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/download_complete_results')
def download_complete_results():
    """Download complete results."""
    path = os.path.join(RESULT_FOLDER, 'new_data_output.csv')
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name='complete_results.csv')
    return jsonify({"error": "Results file not found"}), 404

@app.route('/download_matched_names')
def download_matched_names():
    """Download matched names."""
    path = os.path.join(SEPARATED_FOLDER, 'predicted_1.csv')
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name='matched_names.csv')
    return jsonify({"error": "Matched names file not found"}), 404

@app.route('/download_unmatched_names')
def download_unmatched_names():
    """Download unmatched names."""
    path = os.path.join(SEPARATED_FOLDER, 'predicted_0.csv')
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name='unmatched_names.csv')
    return jsonify({"error": "Unmatched names file not found"}), 404

@app.route('/download_merged_names')
def download_merged_names():
    """Download merged names."""
    path = os.path.join(CLEANED_FOLDER, 'cleaned_predicted_1.csv')
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name='merged_names.csv')
    return jsonify({"error": "Merged names file not found"}), 404


if __name__ == '__main__':
    app.run(debug=True)
