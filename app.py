import streamlit as st
import pandas as pd
from textblob import TextBlob
import joblib
from datetime import datetime
from io import StringIO
import docx2txt
import PyPDF2
import openai
import base64

# Load ML model
model = joblib.load('ensemble_model.pkl')

# Set OpenAI API key securely
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Page config
st.set_page_config(page_title="Article Popularity Predictor", layout="wide")

# Title and layout
st.title("📰 News Article Popularity Predictor")

col1, col2 = st.columns([1, 2])

with col1:
    st.image("images.jpg", use_container_width=True)

with col2:
    st.markdown("Upload your news article and get:")
    st.markdown("• 📈 Popularity Prediction\n• 💡 GPT Suggestions for Improvement")

    # Upload section
    uploaded_file = st.file_uploader("📄 Upload .txt, .docx, or .pdf", type=["txt", "docx", "pdf"])
    publish_date = st.date_input("🗓️ Select expected publish date", value=datetime.now().date())

    def extract_text(file):
        if file.name.endswith(".txt"):
            return StringIO(file.getvalue().decode("utf-8")).read()
        elif file.name.endswith(".docx"):
            return docx2txt.process(file)
        elif file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(file)
            return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        return None

    def get_title_and_headline(text):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        title = lines[0] if lines else ""
        headline = ""
        if len(lines) > 1:
            for i in range(1, len(lines)):
                if len(lines[i].split()) > 5:
                    headline = lines[i]
                    break
        return title, headline

    def get_gpt_suggestions(title, headline):
        prompt = f"Here is a news article:\n\nTitle: {title}\n\nHeadline: {headline}\n\nPlease suggest improvements to make it more engaging and likely to go viral:"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"⚠️ GPT suggestion failed: {e}"

    if uploaded_file:
        content = extract_text(uploaded_file)
        if content:
            title, headline = get_title_and_headline(content)

            st.subheader("📝 Extracted Article Content")
            st.markdown(f"**Title:** {title}")
            st.markdown(f"**Headline:** {headline}")

            # Feature engineering
            sentiment_title = TextBlob(title).sentiment.polarity
            sentiment_headline = TextBlob(headline).sentiment.polarity

            publish_year = publish_date.year
            publish_month = publish_date.month
            publish_day = publish_date.day
            publish_day_of_week = publish_date.weekday()
            publish_hour = 12  # fixed value (assumption)

            input_df = pd.DataFrame([{
                'PublishYear': publish_year,
                'PublishMonth': publish_month,
                'PublishDay': publish_day,
                'PublishDayOfWeek': publish_day_of_week,
                'PublishHour': publish_hour,
                'SentimentTitle': sentiment_title,
                'SentimentHeadline': sentiment_headline
            }])

            if st.button("🚀 Predict Popularity"):
                prediction = model.predict(input_df)[0]
                label_map = {
                    0: "🟥 Unpopular",
                    1: "🟨 Moderately Popular",
                    2: "🟩 Very Popular"
                }
                st.success(f"🎯 Predicted Popularity: {label_map.get(prediction, 'Unknown')}")

                with st.spinner("💬 Asking ChatGPT for improvement suggestions..."):
                    suggestions = get_gpt_suggestions(title, headline)
                    st.subheader("💡 GPT Suggestions to Improve Your Article")
                    st.markdown(suggestions)
        else:
            st.warning("❌ Failed to extract text from uploaded file.")

# Footer
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
            The tool uses sentiment analysis and a trained Random Forest model to predict 
            how popular a news article might be. It also leverages ChatGPT to suggest improvements for virality.
        </p>
        <p>📧 Contact: <a href="mailto:2023bse028@std.must.ac.ug">2023bse028@std.must.ac.ug</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
