import pandas as pd
import numpy as np
import re


# Load the CSV file
input_file = "uploads\dataset_for_training.csv"
df = pd.read_csv(input_file)

# Drop completely empty rows
df.dropna(how='all', inplace=True)

# Define a general cleaner for noise removal
def remove_noise(value):
    if not isinstance(value, str):
        value = str(value)

    value = value.lower().strip()

    # Remove common noise patterns
    value = re.sub(r'(fees?\s*[-/:]?\s*)|(/-)|(\bat clinic\b)|(\bfee\b)|(\bfree\b)', '', value)
    value = re.sub(r'[^\w\d.,@:\s/-]', '', value)  # Keep basic readable characters
    value = value.strip()

    return value

# Apply to all string cells in the DataFrame
df = df.applymap(lambda x: remove_noise(x) if pd.notna(x) else x)

# Save cleaned result
df.to_csv("cleaned/cleaned.csv", index=False)


