# Name Matching System

A web-based system for matching and comparing names using machine learning and various string similarity metrics.

## Features

- Web interface for easy file upload and processing
- Multiple similarity metrics:
  - TF-IDF cosine similarity
  - SBERT embeddings
  - Jaro-Winkler similarity
  - Levenshtein distance
  - Soundex and Metaphone phonetic matching
  - Token set ratio
  - Longest common substring
  - Jaccard similarity
  - N-gram overlap
- XGBoost classifier for final predictions
- Progress feedback and error handling
- Automatic results download

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train.py
```

4. Run the web application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Prepare your data:
   - Training CSV should contain columns: 'name1', 'name2', 'is_match'
   - Test CSV should contain columns: 'name1', 'name2'

2. Upload files:
   - Click "Choose File" to select your training and test CSV files
   - Click "Process Files" to start the matching process

3. Results:
   - The system will process the files and automatically download the results
   - Results CSV will contain original columns plus:
     - 'predicted_match': 1 if names match, 0 if they don't
     - 'match_probability': Probability of match (0-1)

## File Structure

- `app.py`: Flask web application
- `train.py`: Model training and feature computation
- `templates/`: HTML templates
- `uploads/`: Temporary storage for uploaded files
- `results/`: Storage for processed results
- `models/`: Storage for trained models

## Requirements

See `requirements.txt` for full list of dependencies. 