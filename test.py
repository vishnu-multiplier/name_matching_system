import pandas as pd
import numpy as np
import joblib
import logging
import jellyfish
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from fuzzywuzzy import fuzz
from tqdm import tqdm
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sentence_transformers import SentenceTransformer
from train import clean_name, compute_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tqdm.pandas()

# ----------- Utility Functions -----------

def longest_common_substring(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    result = 0
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
                result = max(result, dp[i+1][j+1])
    return result

def jaccard_similarity(a, b):
    set1, set2 = set(a.split()), set(b.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def ngram_overlap(a, b, n=3):
    def ngrams(s):
        return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else set()
    ng1, ng2 = ngrams(a), ngrams(b)
    if not ng1 or not ng2:
        return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)

# ----------- Feature Computation Function -----------

def compute_features(df, tfidf, sbert):
    print("🧹 Cleaning names...")
    name1_clean = df['name1'].progress_apply(clean_name)
    name2_clean = df['name2'].progress_apply(clean_name)

    print("📊 Computing TF-IDF cosine similarity...")
    tfidf1 = tfidf.transform(name1_clean)
    tfidf2 = tfidf.transform(name2_clean)
    cosine_sim = [cosine_similarity(tfidf1[i], tfidf2[i])[0][0] for i in range(tfidf1.shape[0])]

    print("🧠 Computing SBERT cosine similarity...")
    embeds1 = normalize(sbert.encode(name1_clean.tolist(), show_progress_bar=True))
    embeds2 = normalize(sbert.encode(name2_clean.tolist(), show_progress_bar=True))
    sbert_sim = np.einsum('ij,ij->i', embeds1, embeds2)

    print("🔢 Calculating other string similarity features...")
    jaro_sim = df.progress_apply(lambda row: jellyfish.jaro_winkler_similarity(clean_name(row['name1']), clean_name(row['name2'])), axis=1)
    levenshtein = df.progress_apply(lambda row: jellyfish.levenshtein_distance(clean_name(row['name1']), clean_name(row['name2'])), axis=1)
    first_letter_match = (name1_clean.str[0] == name2_clean.str[0]).astype(int)
    len_diff = (name1_clean.str.len() - name2_clean.str.len()).abs()
    soundex_match = df.progress_apply(lambda row: int(jellyfish.soundex(clean_name(row['name1'])) == jellyfish.soundex(clean_name(row['name2']))), axis=1)
    metaphone_match = df.progress_apply(lambda row: int(jellyfish.metaphone(clean_name(row['name1'])) == jellyfish.metaphone(clean_name(row['name2']))), axis=1)
    token_set_ratio = df.progress_apply(lambda row: fuzz.token_set_ratio(clean_name(row['name1']), clean_name(row['name2'])) / 100.0, axis=1)
    lcs = df.progress_apply(lambda row: longest_common_substring(clean_name(row['name1']), clean_name(row['name2'])), axis=1)
    jaccard = df.progress_apply(lambda row: jaccard_similarity(clean_name(row['name1']), clean_name(row['name2'])), axis=1)
    ngram = df.progress_apply(lambda row: ngram_overlap(clean_name(row['name1']), clean_name(row['name2'])), axis=1)

    features = pd.DataFrame({
        'name1_clean': name1_clean,
        'name2_clean': name2_clean,
        'cosine_sim': cosine_sim,
        'sbert_sim': sbert_sim,
        'jaro_sim': jaro_sim,
        'levenshtein': levenshtein,
        'first_letter_match': first_letter_match,
        'len_diff': len_diff,
        'soundex_match': soundex_match,
        'metaphone_match': metaphone_match,
        'token_set_ratio': token_set_ratio,
        'lcs': lcs,
        'jaccard': jaccard,
        'ngram_overlap': ngram
    })

    df = pd.concat([df, features], axis=1)
    return df

# ----------- Main Execution -----------

def load_models():
    """Load all required models for prediction."""
    try:
        logger.info("Loading models...")
        model = joblib.load("models/xgb_model.pkl")
        tfidf = joblib.load("models/tfidf_vectorizer.pkl")
        sbert = joblib.load("models/sbert_model.pkl")
        features = joblib.load("models/features_used.pkl")
        logger.info("Models loaded successfully")
        return model, tfidf, sbert, features
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def predict_matches(df, model, tfidf, sbert, features):
    """Predict matches for a given DataFrame."""
    try:
        logger.info("Computing features for prediction...")
        df = compute_features(df, tfidf, sbert)
        
        logger.info("Making predictions...")

        X = df[features].values  
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        df['predicted_match'] = predictions
        df['match_probability'] = probabilities
        
        return df
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

def separate_results(df):
    """Separate results into matched and unmatched names."""
    
    try:
        logger.info("Separating results...")
        matched = df[df['predicted_match'] == 1].copy()
        unmatched = df[df['predicted_match'] == 0].copy()
        message = (f"Found {len(matched)} matches and {len(unmatched)} non-matches")
        return matched, unmatched, message
    except Exception as e:
        logger.error(f"Error separating results: {e}")
        raise

def main():
    try:
        # Load test data
        logger.info("Loading test data...")
        df = pd.read_csv("new_data.csv")
        
        # Load models
        model, tfidf, sbert, features = load_models()
        
        # Make predictions
        results_df = predict_matches(df, model, tfidf, sbert, features)
        
        # Save complete results
        logger.info("Saving complete results...")
        results_df.to_csv("new_data_output.csv", index=False)
        
        # Separate and save matched/unmatched results
        matched, unmatched, message = separate_results(results_df)
        matched.to_csv("predicted_1.csv", index=False)
        unmatched.to_csv("predicted_0.csv", index=False)
        
        logger.info("Results saved successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
