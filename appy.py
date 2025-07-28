import nltk
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pandas as pd
from textblob import TextBlob
import joblib
import docx2txt
import PyPDF2
import re
import string
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import time

nltk.data.path.append('/home/atahoronnie/nltk_data')

# --- Configure OpenRouter Gemini 2.0 Flash ---
OPENROUTER_API_KEY = st.secrets["openrouter_api_key"]
SITE_URL = st.secrets.get("site_url", "")
SITE_TITLE = st.secrets.get("site_title", "News Predictor App")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

MODEL_NAME = "openai/gpt-3.5-turbo"

@st.cache_resource(show_spinner=False)
def load_models():
    model = joblib.load("lightgbm_popularity.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, label_encoder = load_models()

@st.cache_data(ttl=3600)
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
        pattern_keywords = ["/news/articles/", "/news/live/", "/news/videos/"]
        anchors2 = soup.find_all("a", href=True)
        for anchor in anchors2:
            href = anchor.get("href")
            if any(pat in href for pat in pattern_keywords):
                headline = anchor.get_text(strip=True)
                if headline and headline not in articles:
                    articles.append(headline)
    except Exception as e:
        st.warning(f"Could not fetch existing articles: {e}")
    return articles


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

# --- MAIN UI ---
st.set_page_config(page_title="News Predictor App", layout="wide")

col1, col2 = st.columns([1, 2])

with col1:
    st.image("image.jpg", use_container_width=True)

with col2:
    st.title("\U0001F4F0 News Article Popularity Predictor")
    st.markdown("""
    Welcome to the **News Article Popularity Predictor**!  
    Upload your news article and discover how popular it might be *before publishing*.  
    This app uses:
    - \U0001F9E0 Sentiment analysis
    - \U0001F522 Text feature engineering
    - \U0001F4C8 A trained **LightGBM model**

    ...to predict article popularity and provide **AI-powered advice** to help you improve it.
    """)

st.markdown("### \U0001F4E4 Upload Your News Article (.docx or .pdf)")
uploaded_file = st.file_uploader("", type=["docx", "pdf"])

if uploaded_file is not None:
    raw_text = ""
    if uploaded_file.name.endswith(".docx"):
        raw_text = docx2txt.process(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        raw_text = "\n".join([page.extract_text() for page in reader.pages])
    else:
        st.error("Unsupported file type.")

    if raw_text:
        scraped_articles = fetch_scraped_articles()
        similarity_blocked = False
        if scraped_articles:
            corpus = scraped_articles + [raw_text]
            vectorizer = TfidfVectorizer().fit_transform(corpus)
            similarity_matrix = cosine_similarity(vectorizer[-1], vectorizer[:-1])
            max_similarity = similarity_matrix.max()
            similarity_percent = round(max_similarity * 100, 2)
            st.info(f"Similarity to existing articles: {similarity_percent}%")
            if similarity_percent >= 35.0:
                st.error(f"\u274C This article is {similarity_percent}% similar to existing BBC News articles.")
                similarity_blocked = True

        if not similarity_blocked:
            lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
            title = lines[0] if len(lines) > 0 else ""
            subtitle = lines[1] if len(lines) > 1 else ""
            content = " ".join(lines[2:]) if len(lines) > 2 else ""

            st.subheader("Extracted Title")
            st.write(title)
            if subtitle:
                st.subheader("Extracted Subtitle")
                st.write(subtitle)
            st.subheader("Extracted Content Snippet")
            st.write(content[:500] + "...")

            features_df = extract_features(title, content)
            st.subheader("Extracted Feature Table")
            st.dataframe(features_df.T, use_container_width=True)

            prediction = model.predict(features_df)[0]
            st.success(f"\U0001F4C8 Predicted Popularity: **{prediction.upper()}**")

            advice_prompt = f"You are an expert news editor. Here is a table of extracted features for an uploaded news article. Give 3 specific, actionable suggestions to improve its popularity, based on the table and the full article text.\n\nFeatures Table:\n{features_df.T.to_markdown()}\n\nTitle: {title}\nFull Article:\n{raw_text[:1500]}..."
            with st.spinner("Generating advice..."):
                advice_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are an expert news editor."},
                        {"role": "user", "content": advice_prompt}
                    ]
                )
                st.subheader("Expert Advice")
                st.write(advice_response.choices[0].message.content)

            user_question = st.text_input("Ask a question about your article:")
            if user_question:
                chat_prompt = f"You are a helpful news article assistant.\n\nFeatures Table:\n{features_df.T.to_markdown()}\n\nTitle: {title}\nArticle:\n{raw_text[:1500]}...\n\nUser Question: {user_question}"
                with st.spinner("Thinking..."):
                    chat_response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are a helpful news article assistant."},
                            {"role": "user", "content": chat_prompt}
                        ]
                    )
                    st.write(chat_response.choices[0].message.content)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
        <h4>About Us</h4>
        <p>
            This News Article Popularity Predictor was developed by 
            <strong>ATAHO Ronnie</strong> as part of a machine learning project at 
            Mbarara University of Science and Technology.
        </p>
        <p>
            The tool uses sentiment analysis and a trained LightGBM model to predict 
            how popular a news article might be. It also leverages AI to suggest improvements for virality.
        </p>
        <p>\u2709\ufe0f Contact: <a href='mailto:2023bse028@std.must.ac.ug'>2023bse028@std.must.ac.ug</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
