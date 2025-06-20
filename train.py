import pandas as pd
import numpy as np
import jellyfish
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from sklearn.preprocessing import normalize
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from utility import clean_name, jaccard_similarity,is_complete_overlap_with_empty, same_gender, is_abbreviation

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tqdm.pandas()


def compute_features(df, tfidf, sbert):
    logger.info("Computing features...")
    
    df['name1_clean'] = df['name1'].progress_apply(clean_name)
    df['name2_clean'] = df['name2'].progress_apply(clean_name)

    # TF-IDF cosine similarity
    logger.info("Computing TF-IDF features...")
    tfidf1 = tfidf.transform(df['name1_clean'])
    tfidf2 = tfidf.transform(df['name2_clean'])
    df['cosine_sim'] = [cosine_similarity(tfidf1[i], tfidf2[i])[0][0] for i in range(tfidf1.shape[0])]

    # SBERT embeddings cosine similarity
    logger.info("Computing SBERT embeddings...")
    embeds1 = normalize(sbert.encode(df['name1_clean'].tolist(), show_progress_bar=True))
    embeds2 = normalize(sbert.encode(df['name2_clean'].tolist(), show_progress_bar=True))
    df['sbert_sim'] = np.einsum('ij,ij->i', embeds1, embeds2)

    # Other string similarity features
    logger.info("Computing string similarity metrics...")
    df['jaro_sim'] = df.progress_apply(lambda row: jellyfish.jaro_winkler_similarity(row['name1_clean'], row['name2_clean']), axis=1)
    df['metaphone_match'] = df.progress_apply(lambda row: int(jellyfish.metaphone(row['name1_clean']) == jellyfish.metaphone(row['name2_clean'])), axis=1)
    df['Levenshtein'] = df.progress_apply(lambda row: jellyfish.levenshtein_distance(row['name1_clean'], row['name2_clean']), axis=1)
    df['token_set_ratio'] = df.progress_apply(lambda row: fuzz.token_set_ratio(row['name1_clean'], row['name2_clean']) / 100.0, axis=1)
    df['jaccard'] = df.progress_apply(lambda row: jaccard_similarity(row['name1_clean'], row['name2_clean']), axis=1)
    df['complete_overlap'] = df.progress_apply(lambda row: is_complete_overlap_with_empty(row['name1_clean'], row['name2_clean']), axis=1)
    df['similar_gender'] = df.progress_apply(lambda row: same_gender(row['name1_clean'], row['name2_clean']),axis=1)
    df['is_abbreviation'] = df.progress_apply(lambda row: int(is_abbreviation(row['name1_clean'], row['name2_clean'])), axis=1)


    return df

def prepare_data(df):
    """Prepare data for training by computing features."""
    logger.info("Preparing data...")
    
    # Initialize models
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    tfidf.fit(pd.concat([df['name1'], df['name2']]).dropna().apply(clean_name))
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute features
    df = compute_features(df, tfidf, sbert)

    # Select features
    feature_cols = [
        'cosine_sim', 'sbert_sim','Levenshtein', 'jaro_sim',
        'metaphone_match', 'token_set_ratio', 'jaccard',
        'complete_overlap','similar_gender','is_abbreviation'
    ]

    X = df[feature_cols]
    y = df['is_match']

    # Store the original names in the index
    X.index = pd.MultiIndex.from_arrays([df['name1'], df['name2']], names=['name1', 'name2'])

    return X, y, tfidf, sbert  # Return all necessary components

def train_model(X_train, y_train, tfidf, sbert):
    """Train the model and return model components."""
    logger.info("Training model...")
    
    # Train XGBoost model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    return model  # Return only the trained model

if __name__ == "__main__":
    try:
        logger.info("Loading data...")
        df = pd.read_csv("dataset_for_training.csv")
        df['is_match'] = df['is_match'].astype(int)

        # Prepare data and get fitted tfidf and sbert models
        X, y, tfidf, sbert = prepare_data(df)

        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Train XGBoost model
        model = train_model(X_train, y_train, tfidf, sbert)

        # Evaluate
        y_pred = model.predict(X_val)
        logger.info(f"Accuracy: {accuracy_score(y_val, y_pred)}")
        logger.info(f"\nClassification Report:\n{classification_report(y_val, y_pred)}")

        # Save all models and feature list
        logger.info("Saving models...")
        joblib.dump(model, "xgb_model.pkl")
        joblib.dump(tfidf, "tfidf_vectorizer.pkl")
        joblib.dump(sbert, "sbert_model.pkl")
        joblib.dump(X.columns.tolist(), "features_used.pkl")
        logger.info("Models saved successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
