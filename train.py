import pandas as pd
import numpy as np
import re
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tqdm.pandas()

def clean_name(name):
    """Cleans and normalizes names by removing titles, professions, and location phrases."""
    if pd.isnull(name):
        return ""

    name = name.lower()
    name = re.sub(r'\.(?=\w)', '. ', name)

    # Step 1: Remove common name prefixes
    prefixes = {'dr', 'mr', 'mrs', 'ms', 'miss', 'prof', 'sir', 'madam', 'shri', 'smt', 'doctor', 'professor'}
    words = name.split()
    while words and words[0].rstrip('.') in prefixes:
        words.pop(0)
    name = ' '.join(words)

    # Step 2: Remove pattern-based profession + location
    profession_keywords = [
        'doctor', 'surgeon', 'dentist', 'physician', 'consultant',
        'orthopedic', 'cardiologist', 'neurologist', 'pediatrician','pulmonologist',
        'dermatologist', 'psychiatrist', 'ophthalmologist', 'ent specialist',
        'urologist', 'gastroenterologist', 'oncologist', 'gynecologist'
    ]

    # Remove patterns like 'doctor in delhi', 'surgeon from pune'
    for prof in profession_keywords:
        name = re.sub(rf'{prof}\s+(in|from|at)\s+\w+', '', name)
        name = re.sub(rf'{prof}\s+\w+', '', name)  # 'surgeon delhi'
        name = re.sub(rf'\b{prof}\b', '', name)    # lone profession

    # Step 3: Normalize remaining text
    name = re.sub(r'\b([a-z])\.', r'\1', name)  # A. → A
    name = re.sub(r"[^a-z\s'-]", '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    # Step 4: Deduplicate consecutive words
    tokens = name.split()
    final_tokens = [t for i, t in enumerate(tokens) if i == 0 or t != tokens[i - 1]]

    return ' '.join(final_tokens)



def longest_common_substring(s1, s2):
    m = [[0]*(1+len(s2)) for _ in range(1+len(s1))]
    longest = 0
    for i in range(1, 1+len(s1)):
        for j in range(1, 1+len(s2)):
            if s1[i-1] == s2[j-1]:
                m[i][j] = m[i-1][j-1] + 1
                longest = max(longest, m[i][j])
            else:
                m[i][j] = 0
    return longest

def jaccard_similarity(a, b):
    set1, set2 = set(a.split()), set(b.split())
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0

def ngram_overlap(a, b, n=3):
    ngrams = lambda s: {s[i:i+n] for i in range(len(s)-n+1)} if len(s) >= n else set()
    ng1, ng2 = ngrams(a), ngrams(b)
    return len(ng1 & ng2) / len(ng1 | ng2) if ng1 | ng2 else 0.0

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
    df['levenshtein'] = df.progress_apply(lambda row: jellyfish.levenshtein_distance(row['name1_clean'], row['name2_clean']), axis=1)
    df['first_letter_match'] = (df['name1_clean'].str[0] == df['name2_clean'].str[0]).astype(int)
    df['len_diff'] = (df['name1_clean'].str.len() - df['name2_clean'].str.len()).abs()
    df['soundex_match'] = df.progress_apply(lambda row: int(jellyfish.soundex(row['name1_clean']) == jellyfish.soundex(row['name2_clean'])), axis=1)
    df['metaphone_match'] = df.progress_apply(lambda row: int(jellyfish.metaphone(row['name1_clean']) == jellyfish.metaphone(row['name2_clean'])), axis=1)
    df['token_set_ratio'] = df.progress_apply(lambda row: fuzz.token_set_ratio(row['name1_clean'], row['name2_clean']) / 100.0, axis=1)
    df['lcs'] = df.progress_apply(lambda row: longest_common_substring(row['name1_clean'], row['name2_clean']), axis=1)
    df['jaccard'] = df.progress_apply(lambda row: jaccard_similarity(row['name1_clean'], row['name2_clean']), axis=1)
    df['ngram_overlap'] = df.progress_apply(lambda row: ngram_overlap(row['name1_clean'], row['name2_clean']), axis=1)

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
        'cosine_sim', 'sbert_sim', 'jaro_sim', 'levenshtein',
        'first_letter_match', 'len_diff', 'soundex_match', 'lcs',
        'metaphone_match', 'token_set_ratio', 'jaccard', 'ngram_overlap'
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
        use_label_encoder=False,
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
