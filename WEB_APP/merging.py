import pandas as pd
from urllib.parse import urlparse
import os

def download_merged_names():
    input_path = os.path.join('seperated', 'predicted_1.csv')
    output_path = os.path.join('cleaned', 'cleaned_predicted_1.csv')

    priority_order = [
        "practo.com", "lybrate.com", "apollo247", "bajajfinservhealth", "myupchar.com",
        "medibuddy.in", "credihealth.com", "skedoc.com", "doctoriduniya.com", "sehat.com",
        "deldure.com", "ask4healthcare", "hexahealth.com", "meddco.com", "lazoi.com",
        "quickerala.com", "patakare.com", "docindia.org", "mymedisage.com", "drlogy.com",
        "doctor360", "ihindustan.com", "healthfrog", "www.drdata", "prescripson.com",
        "curofy.com", "justdialdds.com", "converse.rgcross.com", "healthworldhospitals.com",
        "healthgrades.com"
    ]
    priority_map = {domain: i for i, domain in enumerate(priority_order)}
    default_priority = len(priority_order) + 1

    def extract_domain(url):
        try:
            return urlparse(url).netloc.replace("www.", "")
        except:
            return ""

    df = pd.read_csv(input_path)
    df['domain'] = df['url'].apply(extract_domain)
    df['priority'] = df['domain'].apply(lambda d: priority_map.get(d, default_priority))
    df['missing_count'] = df.isnull().sum(axis=1)

    merged_rows = []
    for record_id, group in df.groupby('record_id'):
        group = group.sort_values(by=['domain', 'missing_count']).drop_duplicates(subset='domain', keep='first')
        group = group.sort_values(by='priority').reset_index(drop=True)

        merged_row = group.iloc[0].copy()
        contributing_urls = set()
        base_url = merged_row['url']

        for _, row in group.iterrows():
            used = False
            for col in df.columns:
                if col not in ['record_id', 'url', 'domain', 'priority', 'missing_count']:
                    if pd.isna(merged_row[col]) or merged_row[col] == '':
                        if not pd.isna(row[col]) and row[col] != '':
                            merged_row[col] = row[col]
                            used = True
            if used and row['url'] != base_url:
                contributing_urls.add(row['url'])

        contributing_urls = sorted(contributing_urls, key=lambda url: priority_map.get(extract_domain(url), default_priority))
        for i, extra_url in enumerate(contributing_urls, start=2):
            merged_row[f"url_{i}"] = extra_url

        merged_rows.append(merged_row)

    final_df = pd.DataFrame(merged_rows)

    columns_to_remove = [
        'cosine_sim', 'sbert_sim', 'jaro_sim', 'levenshtein', 'first_letter_match',
        'len_diff', 'soundex_match', 'metaphone_match', 'token_set_ratio',
        'lcs', 'jaccard', 'ngram_overlap', 'name1_clean', 'name2_clean',
        'domain', 'priority', 'missing_count', 'predicted_match'
    ]
    final_df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
    final_df = final_df.dropna(how='all')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    output = download_merged_names()
    print("Output saved at:", output)