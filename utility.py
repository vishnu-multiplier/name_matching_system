
import pandas as pd
import re



def attempt_safe_split(word):
    """Try to split joined words based on vowel-consonant or mid-point patterns."""
    if len(word) < 7 or ' ' in word:
        return word

    # Heuristic: Try mid-point split (best-effort)
    mid = len(word) // 2
    for offset in range(1, 4):
        i = mid - offset
        if word[i].isalpha() and word[i+1].isalpha():
            return word[:i+1] + ' ' + word[i+1:]
    return word

def clean_name(name):
    """Cleans and normalizes names, and splits joined names like 'ravikumar' -> 'ravi kumar' without using name lists."""
    if pd.isnull(name):
        return ""

    name = name.lower()
    name = re.sub(r'\.(?=\w)', '. ', name)

    # Step 1: Remove common prefixes
    prefixes = {'dr', 'mr', 'mrs', 'ms', 'miss', 'prof', 'sir', 'madam', 'shri', 'smt', 'doctor', 'professor'}
    words = name.split()
    while words and words[0].rstrip('.') in prefixes:
        words.pop(0)
    name = ' '.join(words)

    # Step 2: Remove profession + location patterns
    profession_keywords = [
        'doctor', 'surgeon', 'dentist', 'physician', 'consultant',
        'orthopedic', 'cardiologist', 'neurologist', 'pediatrician', 'pulmonologist',
        'dermatologist', 'psychiatrist', 'ophthalmologist', 'ent specialist',
        'urologist', 'gastroenterologist', 'oncologist', 'gynecologist'
    ]
    for prof in profession_keywords:
        name = re.sub(rf'{prof}\s+(in|from|at)\s+\w+', '', name)
        name = re.sub(rf'{prof}\s+\w+', '', name)
        name = re.sub(rf'\b{prof}\b', '', name)

    # Step 3: Normalize and clean
    name = re.sub(r'\b([a-z])\.', r'\1', name)
    name = re.sub(r"[^a-z\s'-]", '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    # Step 4: Attempt safe split of joined names (e.g., "ravikumar" -> "ravi kumar")
    tokens = name.split()
    split_tokens = []
    for token in tokens:
        if len(token) >= 7 and ' ' not in token:
            split_token = attempt_safe_split(token)
            split_tokens.extend(split_token.split())
        else:
            split_tokens.append(token)

    # Step 5: Deduplicate consecutive words
    final_tokens = [t for i, t in enumerate(split_tokens) if i == 0 or t != split_tokens[i - 1]]

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

def is_complete_overlap_with_empty(name1, name2):
    if not name1 or not name2:
        return 0

    name1 = name1.strip().lower()
    name2 = name2.strip().lower()

    # Check exact containment
    if name1 in name2:
        leftover = name2.replace(name1, "").strip()
        if not leftover:
            return 1
        if not name1.replace(leftover, "").strip():
            return 1
    elif name2 in name1:
        leftover = name1.replace(name2, "").strip()
        if not leftover:
            return 1
        if not name2.replace(leftover, "").strip():
            return 1

    return 0




def is_valid_name(name):
    """Valid name is at least 2 characters (to exclude initials)."""
    return len(name.strip()) >= 2

def is_female_name(name):
    """True if name ends in 'a' or 'i' (and is not an initial)."""
    name = name.lower().strip()
    return is_valid_name(name) and (name.endswith('a') or name.endswith('i'))

def get_gender_label(name_parts):
    """Returns 'female', 'male', or 'mixed'."""
    female_count = sum(1 for name in name_parts if is_female_name(name))
    
    if female_count == len(name_parts):
        return 'female'
    elif female_count == 0:
        return 'male'
    else:
        return 'mixed'

def same_gender(name_str1, name_str2):
    parts1 = [n.strip().lower() for n in name_str1.split() if is_valid_name(n)]
    parts2 = [n.strip().lower() for n in name_str2.split() if is_valid_name(n)]
    
    set1 = set(parts1)
    set2 = set(parts2)
    common_parts = set1.intersection(set2)

    # Rule 1: Shared name indicates same gender
    for name in common_parts:
        if is_female_name(name) or not is_female_name(name):
            return 1  # same person or same gender

    # Rule 2: No common parts, use gender logic
    gender1 = get_gender_label(parts1)
    gender2 = get_gender_label(parts2)

    if 'mixed' in (gender1, gender2) or 'unknown' in (gender1, gender2):
        return 0
    return 1 if gender1 == gender2 else 0

def is_abbreviation(name1, name2):
    def get_initials(tokens):
        return set(token[0] for token in tokens if len(token) > 0 and token not in {'a', 'i'})

    tokens1 = name1.split()
    tokens2 = name2.split()

    short, long = (tokens1, tokens2) if len(tokens1) <= len(tokens2) else (tokens2, tokens1)

    if not (2 <= len(short) <= 4):
        return 0

    short_initials = get_initials(short)
    long_initials = get_initials(long)

    return int(short_initials.issubset(long_initials))
