# ChatGPT Reviews — Text Mining & Sentiment Analysis

> User Perception Analysis of ChatGPT Using Text Mining Techniques — a full NLP pipeline applied to 100,000 Google Play Store reviews.

<p align="center">
  <a href="https://github.com/cayirhalil/chatgpt-perception-text-mining"><img alt="Repo" src="https://img.shields.io/badge/GitHub-Repository-1f2937?logo=github"></a>
  <a href="https://cayirhalil.github.io/chatgpt-perception-text-mining"><img alt="Pages" src="https://img.shields.io/badge/Live-Showcase-22d3ee"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-22c55e">
</p>

---

## Overview

This project applies a complete text mining pipeline to a large corpus of user reviews of the
ChatGPT Android application, scraped from the Google Play Store. The goal is to understand how
users perceive AI tools at scale by combining **supervised sentiment classification** with
**unsupervised topic modeling** and **temporal trend analysis**.

It was developed as the semester project for the **Text Mining** course at Mendel University in Brno,
Faculty of Business and Economics, in the academic year 2025/2026.

**Research question:** What sentiment and discussion topics characterise user feedback on ChatGPT,
and how do they evolve over time?

---

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle — Google Play Store reviews of the ChatGPT app |
| Raw size | 100,000 reviews, 8 columns |
| Columns | `Name`, `Rating`, `Comment`, `Date`, `Country`, `Thumbs Up`, `Review ID`, `App Version` |
| Rating distribution | Heavily skewed positive — 74.4% five-star, 6.9% one-star |
| Region | United States |
| Date range (after cleaning) | November 2023 – July 2024 (9 months) |
| Cleaned size | 32,272 reviews |

After preprocessing the corpus shrank from 100,000 to **32,272** reviews: null comments (1,467),
very short comments under 10 characters (20,486), and duplicates (15,525) were removed, followed by
class-imbalance correction.

---

## Methodology

The pipeline follows the standard text mining steps taught in the course.

1. **Data Collection** — Kaggle dataset of 100,000 Google Play reviews.
2. **Preprocessing** — null and short-comment removal, deduplication, tokenization, stopword removal
   and lemmatization with spaCy (`en_core_web_sm`); emoji detection (13.1% of reviews contained emojis).
3. **EDA** — rating distribution, comment length, emoji-vs-rating analysis, review volume over time.
4. **Feature Extraction** — TF-IDF vectorization with unigrams + bigrams (`ngram_range=(1,2)`,
   `max_features=10,000`, `min_df=3`).
5. **Modeling**
   - *Sentiment classification:* Logistic Regression vs. Multinomial Naive Bayes
     (Rating → label: 1–2 negative, 3 neutral, 4–5 positive). Class imbalance handled by undersampling.
   - *Topic modeling:* Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF),
     evaluated with UMass coherence.
6. **Evaluation & Interpretation** — macro F1, confusion matrices, topic coherence, per-topic mean
   rating, and temporal sentiment/topic trends.

---

## Results

> Headline metrics from the current run. Update the numbers here and in `index.html` as the analysis is refined.

| Metric | Value |
|---|---|
| Reviews analyzed (cleaned) | 32,272 |
| Best sentiment model | Logistic Regression + TF-IDF (bigrams) |
| Macro F1 (3-class) | 0.5915 |
| Sentiment split | Positive 66.9% · Negative 22.3% · Neutral 10.8% |
| Topics discovered (LDA) | 6 |
| Dominant negative topic | Technical Issues & Errors — 24.5% of reviews, mean rating 2.87 |
| Dominant positive topic | Homework & Study Help — 25.2% of reviews, mean rating 4.32 |

**Key finding:** mean satisfaction declined gradually across the observation window. As the user base
expanded, technical-complaint topics grew fastest — consistent with known scaling challenges of AI services.

---

## Tech Stack

- **Language:** Python 3.10+
- **Data:** pandas, numpy
- **NLP:** NLTK, spaCy (`en_core_web_sm`)
- **ML / Topic Modeling:** scikit-learn (TF-IDF, Logistic Regression, Naive Bayes, LDA, NMF)
- **Visualization:** matplotlib, seaborn, wordcloud
- **Optional:** transformers (transformer-based sentiment baseline)

---

## Repository Structure

```
chatgpt-perception-text-mining/
├── data/
│   └── GPT_reviews.csv            # raw dataset (not tracked — see .gitignore)
├── src/
│   ├── 01_preprocessing.py        # cleaning, tokenization, lemmatization
│   ├── 02_data_cleaning.py        # deduplication, class balancing
│   ├── 03_sentiment_classification.py  # TF-IDF + LR/NB
│   ├── 04_topic_modeling.py       # LDA / NMF + coherence
│   ├── 05_visualization.py        # wordclouds, charts
│   └── 06_temporal_analysis.py    # trends over time
├── plots/                         # generated figures
├── report/
│   └── text_mining_report.docx    # final written report
├── index.html                     # project showcase page
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting Started

```bash
# clone
git clone https://github.com/cayirhalil/chatgpt-perception-text-mining.git
cd chatgpt-perception-text-mining

# environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# place GPT_reviews.csv in data/ then run the pipeline
python src/01_preprocessing.py
python src/03_sentiment_classification.py
python src/04_topic_modeling.py
```

---

## Team

| Member | Role |
|---|---|
| **Halil Can Cayir** | Preprocessing, sentiment modeling, showcase |
| **Umut Sedat Efeoglu** | Topic modeling, evaluation |
| **Nurlan Gasimzade** | EDA, visualization, reporting |

**Supervisor:** doc. Ing. František Dařena, Ph.D.
**Institution:** Mendel University in Brno · Faculty of Business and Economics
**Course:** Text Mining · Academic year 2025/2026

---

## License

Released under the MIT License. The dataset is the property of its original Kaggle authors and is used
here for academic purposes only.
