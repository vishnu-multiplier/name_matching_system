# Name Matching System

A secure, web-based system that intelligently compares and matches names using machine learning and multiple string similarity algorithms. Built to streamline entity resolution tasks for test and training data with instant feedback and downloadable results.

> ✅ **Created by Banavath Vishnu for Multiplier AI**

---

## 🚀 Features

* **Interactive Web Interface**
  Upload CSV files, view detailed similarity metrics, and download results effortlessly.

* **Dual Dataset Support**

  * **Test Dataset Tab**: For prediction
  * **Training Dataset Tab**: For exploring training data (requires login)

* **Rich Similarity Metrics**

  * TF-IDF Cosine Similarity
  * SBERT Embeddings
  * Jaro-Winkler Distance
  * Levenshtein Distance
  * Soundex & Metaphone
  * Token Set Ratio
  * Longest Common Substring
  * Jaccard Similarity
  * N-gram Overlap

* **Pre-Trained ML Model (XGBoost)**
  Instant match prediction with confidence scores—no training required.

* **Real-time Feedback**
  Progress bar, error handling, and automatic results download.

* **Secure Access for Training Dataset**
  Login-protected interface for accessing and visualizing training data.

---

## 🔐 Training Dataset Access

To view the **Training Dataset** tab:

* **Username**: `admin`
* **Password**: `admin`

---

## ⚙️ Setup Instructions

### 1. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Application

```bash
python app.py
```

Then open your browser at:
👉 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 📄 Usage Guide

### ✅ Test Dataset Format

CSV file **must contain** the following columns:

* `record_id`
* `name1`
* `name2`
* `url`

### ✅ Training Dataset Format

CSV file **must contain**:

* `name1`
* `name2`
* `is_match`

> ⚠️ **Note:** Matching is **strictly case-sensitive and whitespace-sensitive**. Ensure input is clean and standardized.

---

### 📥 How to Use

1. Open the application in your browser.
2. Navigate to the appropriate tab:

   * **Test Dataset** for prediction
   * **Training Dataset** (login required) for inspection
3. Upload your CSV file.
4. Click **“Process Files”**.
5. View similarity metrics and download the results file.

**Output CSV includes**:

* All original columns
* `predicted_match`: 1 or 0
* `match_probability`: Confidence score between 0 and 1

---

## 📁 Project Structure

```
.
├── app.py             # Flask web server
├── templates/         # HTML frontend templates
├── uploads/           # Temporary file storage
├── results/           # Processed output files
├── models/            # Pre-trained model files
└── requirements.txt   # Python dependencies
```

---

## 📦 Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

---

## 👤 Author

Developed by **Banavath Vishnu**
For **Multiplier AI**
[LinkedIn](https://www.linkedin.com/in/banavath-vishnu) • [GitHub](https://github.com/Banavath-Vishnu)

---
