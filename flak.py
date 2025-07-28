# Flask Version of Your Streamlit App

from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import pandas as pd
import numpy as np
import nltk
import os
import docx2txt
import PyPDF2
import re
import string
from textblob import TextBlob
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import requests
from openai import OpenAI

nltk.data.path.append('/home/atahoronnie/nltk_data')

app = Flask(__name__)
app.secret_key = "supersecret"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load models ---
model = joblib.load("lightgbm_popularity.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# --- OpenAI Setup ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
MODEL_NAME = "openai/gpt-3.5-turbo"

# --- Scrape existing articles ---
def fetch_scraped_articles():
    url = "https://www.bbc.com/news"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ArticleScraper/1.0)"}
    articles = []
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        anchors = soup.find_all("a", class_="gs-c-promo-heading")
        for anchor in anchors:
            headline = anchor.get_text(strip=True)
            if headline:
                articles.append(headline)
    except:
        pass
    return articles

# --- Feature extraction ---
def extract_features(title, content):
    features = {}
    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())

    tokens_title = tokenize(title)
    tokens_content = tokenize(content)
    all_tokens = tokens_title + tokens_content

    features['n_tokens_title'] = len(tokens_title)
    features['n_tokens_content'] = len(tokens_content)
    features['n_unique_tokens'] = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
    features['average_token_length'] = np.mean([len(t) for t in all_tokens]) if all_tokens else 0
    features['num_hrefs'] = len(re.findall(r'http[s]?://', content))
    features['num_self_hrefs'] = content.lower().count('read more')
    features['num_imgs'] = content.count('<img') + content.count('![image')
    features['num_videos'] = content.count('<video') + content.count('youtube.com') + content.count('vimeo.com')

    r = Rake()
    r.extract_keywords_from_text(content)
    keywords = r.get_ranked_phrases_with_scores()
    keyword_scores = [score for score, phrase in keywords]
    features['num_keywords'] = len(keywords)
    for key in ['kw_min_min', 'kw_max_min', 'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg']:
        features[key] = np.mean(keyword_scores) if keyword_scores else 0

    lda_dummy = np.random.dirichlet(np.ones(5), size=1)[0]
    for i in range(5):
        features[f'LDA_0{i}'] = lda_dummy[i]

    channel_keywords = {
        'data_channel_is_lifestyle': ['lifestyle', 'health', 'food', 'fashion', 'wellness', 'fitness', 'diet'],
        'data_channel_is_entertainment': ['entertainment', 'music', 'movie', 'film', 'celebrity', 'show', 'concert', 'tv'],
        'data_channel_is_bus': ['business', 'finance', 'economy', 'market', 'stock', 'trade', 'company', 'industry'],
        'data_channel_is_socmed': ['social', 'media', 'facebook', 'twitter', 'instagram', 'tiktok', 'snapchat', 'youtube'],
        'data_channel_is_tech': ['tech', 'technology', 'gadgets', 'software', 'hardware', 'ai', 'robot', 'computer', 'internet'],
        'data_channel_is_world': ['world', 'global', 'international', 'country', 'nation', 'politics', 'government', 'diplomacy']
    }
    text_for_channel = (title + ' ' + content).lower()
    words_set = set(re.findall(r'\b\w+\b', text_for_channel))
    for channel, keywords in channel_keywords.items():
        features[channel] = int(any(kw in words_set for kw in keywords))

    blob = TextBlob(content)
    features['global_subjectivity'] = blob.sentiment.subjectivity
    features['global_sentiment_polarity'] = blob.sentiment.polarity

    words = tokenize(content)
    pos_words = [w for w in words if TextBlob(w).sentiment.polarity > 0]
    neg_words = [w for w in words if TextBlob(w).sentiment.polarity < 0]
    features['global_rate_positive_words'] = len(pos_words) / len(words) if words else 0
    features['global_rate_negative_words'] = len(neg_words) / len(words) if words else 0
    features['rate_positive_words'] = len(set(pos_words)) / len(words) if words else 0
    features['rate_negative_words'] = len(set(neg_words)) / len(words) if words else 0
    features['avg_positive_polarity'] = np.mean([TextBlob(w).sentiment.polarity for w in pos_words]) if pos_words else 0
    features['min_positive_polarity'] = np.min([TextBlob(w).sentiment.polarity for w in pos_words]) if pos_words else 0
    features['max_positive_polarity'] = np.max([TextBlob(w).sentiment.polarity for w in pos_words]) if pos_words else 0
    features['avg_negative_polarity'] = np.mean([TextBlob(w).sentiment.polarity for w in neg_words]) if neg_words else 0
    features['min_negative_polarity'] = np.min([TextBlob(w).sentiment.polarity for w in neg_words]) if neg_words else 0
    features['max_negative_polarity'] = np.max([TextBlob(w).sentiment.polarity for w in neg_words]) if neg_words else 0

    title_blob = TextBlob(title)
    features['title_subjectivity'] = title_blob.sentiment.subjectivity
    features['title_sentiment_polarity'] = title_blob.sentiment.polarity
    features['abs_title_subjectivity'] = abs(title_blob.sentiment.subjectivity)
    features['abs_title_sentiment_polarity'] = abs(title_blob.sentiment.polarity)

    EXPECTED_FEATURES = list(features.keys())
    return pd.DataFrame([features])[EXPECTED_FEATURES]

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction, similarity_msg, advice, question_answer = None, None, None, None
    title, subtitle, content = "", "", ""
    if request.method == 'POST':
        file = request.files['article']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            if filename.endswith(".docx"):
                raw_text = docx2txt.process(filename)
            elif filename.endswith(".pdf"):
                reader = PyPDF2.PdfReader(filename)
                raw_text = "\n".join([page.extract_text() for page in reader.pages])
            else:
                flash("Unsupported file type")
                return redirect(url_for('index'))

            scraped_articles = fetch_scraped_articles()
            similarity_blocked = False
            if scraped_articles:
                corpus = scraped_articles + [raw_text]
                vectorizer = TfidfVectorizer().fit_transform(corpus)
                similarity_matrix = cosine_similarity(vectorizer[-1], vectorizer[:-1])
                max_similarity = similarity_matrix.max()
                similarity_percent = round(max_similarity * 100, 2)
                similarity_msg = f"Similarity to existing BBC articles: {similarity_percent}%"
                if similarity_percent >= 35.0:
                    flash(similarity_msg)
                    return redirect(url_for('index'))

            lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
            title = lines[0] if len(lines) > 0 else ""
            subtitle = lines[1] if len(lines) > 1 else ""
            content = " ".join(lines[2:]) if len(lines) > 2 else ""

            features_df = extract_features(title, content)
            prediction = model.predict(features_df)[0]

            prompt = f"You are an expert news editor. Here is a table of extracted features for an uploaded news article. Give 3 suggestions to improve it.\n\n{features_df.T.to_markdown()}\n\nTitle: {title}\nContent: {raw_text[:1500]}..."
            advice_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert news editor."},
                    {"role": "user", "content": prompt}
                ]
            )
            advice = advice_response.choices[0].message.content

    return render_template("index.html", prediction=prediction, advice=advice, title=title, subtitle=subtitle, content=content[:500])

if __name__ == '__main__':
    app.run(debug=True)
