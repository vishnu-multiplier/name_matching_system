
import pandas as pd
import re

def clean_name(name):
    """Basic cleaning: lowercase, remove prefixes, professions, special characters."""
    if not isinstance(name, str) or pd.isnull(name):
        return ""

    name = name.lower().strip()

    prefixes = {'dr', 'mr', 'mrs', 'ms', 'miss', 'prof', 'sir', 'madam', 'shri', 'smt', 'doctor', 'professor'}
    words = [w for w in name.split() if w.rstrip('.') not in prefixes]

    profession_keywords = (
        'doctor', 'surgeon', 'dentist', 'physician', 'consultant','general',
        'orthopedic', 'cardiologist', 'neurologist', 'pediatrician', 'pulmonologist',
        'dermatologist', 'psychiatrist', 'ophthalmologist', 'ent specialist',
        'urologist', 'gastroenterologist', 'oncologist', 'gynecologist'
    )
    name = ' '.join(words)
    pattern = r'\b(?:' + '|'.join(profession_keywords) + r')\b.*?(?=\b[a-z]+\b|$)'
    name = re.sub(pattern, '', name)

    name = re.sub(r"[^a-z\s]", '', name)
    return re.sub(r'\s+', ' ', name).strip()


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

def same_gender(name1: str, name2: str) -> int:
    tokens1 = [w for w in name1.split() if len(w) > 1]
    tokens2 = [w for w in name2.split() if len(w) > 1]

    common = set(tokens1) & set(tokens2)

    # If there's a common token that ends in 'a' or 'i', likely same gender
    for token in common:
        if token[-1] in {'a', 'i'}:
            return 1

    # If both names have majority of words ending in 'a' or 'i', assume female
    female1 = sum(w[-1] in {'a', 'i'} for w in tokens1)
    female2 = sum(w[-1] in {'a', 'i'} for w in tokens2)

    if female1 > 0 and female2 > 0:
        return 1

    return 0


def is_abbreviation(name1, name2):
    """
    Checks if one name is a valid abbreviation of the other.

    Rules:
      1. Identify which input is the “long” name vs. the “short” name
         by comparing total non-space characters.
      2. Only attempt an abbreviation check if the short name has 2–4 tokens.
      3. For each token in the short name:
         - If it’s one character → match it as an INITIAL (first letter) 
           against an unused word in the long name.
         - If it’s longer than one character → it must match an UNUSED
           word in the long name exactly (full-word match).
      4. No long-name word can be reused for two tokens.
      5. Order of short-name tokens doesn’t matter (jumbled allowed).
      6. Comparison is case-insensitive.

    Returns:
        1 if valid abbreviation, else 0.
    """
    # 1. Normalize & split
    t1 = name1.lower().split()
    t2 = name2.lower().split()

    # 2. Decide long vs. short by total character count
    len1 = sum(len(tok) for tok in t1)
    len2 = sum(len(tok) for tok in t2)
    if len1 >= len2:
        long_tokens, short_tokens = t1, t2
    else:
        long_tokens, short_tokens = t2, t1

    # 3. Enforce token-count constraint
    if not (2 <= len(short_tokens) <= 4):
        return 0

    # 4. Prepare “used” flags so no long word is reused
    used = [False] * len(long_tokens)

    # 5. Match each short token
    for st in short_tokens:
        matched = False

        # full-word case
        if len(st) > 1:
            for i, lt in enumerate(long_tokens):
                if not used[i] and st == lt:
                    used[i] = True
                    matched = True
                    break

        # initial case
        else:  # single character
            for i, lt in enumerate(long_tokens):
                if not used[i] and lt.startswith(st):
                    used[i] = True
                    matched = True
                    break

        if not matched:
            return 0  # fail fast if any token can't match

    return 1

