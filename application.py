import streamlit as st
import pandas as pd
import joblib
from textblob import TextBlob
import datetime
import os # Import os module to check for file existence

# --- Dummy Model for Demonstration ---
# Since 'best_news_classifier.pkl' is not provided, we'll create a simple dummy model.
# In a real scenario, you would train and save your actual scikit-learn model.

class DummyNewsClassifier:
    """
    A simple dummy classifier to simulate predictions for demonstration purposes.
    It predicts '🔥 viral' if sentiment is positive, '❄️ not viral' if negative,
    and '🟡 medium' otherwise.
    This dummy model does NOT use all features, but the actual PyCaret model might.
    """
    def predict(self, X):
        predictions = []
        # The dummy model will now try to use 'Sentiment' if available,
        # but is robust if only 'text' is passed, as per the fix.
        if 'Sentiment' in X.columns:
            for index, row in X.iterrows():
                if row['Sentiment'] > 0.1: # Positive sentiment
                    predictions.append('🔥 viral')
                elif row['Sentiment'] < -0.1: # Negative sentiment
                    predictions.append('❄️ not viral')
                else: # Neutral sentiment
                    predictions.append('🟡 medium')
        elif 'text' in X.columns: # Fallback for dummy model if only text is passed
            for text_content in X['text']:
                sentiment = TextBlob(text_content).sentiment.polarity
                if sentiment > 0.1:
                    predictions.append('🔥 viral')
                elif sentiment < -0.1:
                    predictions.append('❄️ not viral')
                else:
                    predictions.append('🟡 medium')
        else:
            predictions = ['🟡 medium'] * len(X) # Default if no relevant columns
        return predictions

# Check if the actual model file exists, otherwise use the dummy model
model_path = 'best_news_classifier.pkl'
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.sidebar.success(f"Loaded actual model from '{model_path}'")
    except Exception as e:
        st.sidebar.error(f"Error loading actual model: {e}. Using dummy model.")
        st.sidebar.info("If your model was trained with PyCaret, ensure 'pycaret' is installed and compatible.")
        model = DummyNewsClassifier()
else:
    model = DummyNewsClassifier()
    st.sidebar.warning(f"'{model_path}' not found. Using a dummy model for demonstration.")
# --- End Dummy Model ---


# Streamlit app configuration
st.set_page_config(page_title="News Virality Predictor", layout="centered")
st.title("📰 News Article Virality Predictor")
st.markdown("Upload a headline and article to see if it's likely to be 🔥 viral, 🟡 medium, or ❄️ not viral.")

# Input form
heading = st.text_input("📝 Article Heading")
article = st.text_area("📄 Article Content")
date = st.date_input("📅 Publication Date", value=datetime.date.today())

# Predict button
if st.button("🚀 Check Virality"):
    if not heading or not article:
        st.warning("Please provide both a heading and article content.")
    else:
        # Combine heading and article content into a single 'text' string
        full_text = heading + ". " + article

        # --- Feature Engineering (for display/info only, not passed to model directly) ---
        # These features were likely generated internally by your PyCaret model's pipeline
        # from the 'text' column during training.
        # We calculate them here for potential display or debugging, but DO NOT pass them
        # as separate columns to the model's predict method.
        heading_length_display = len(heading)
        sentiment_display = TextBlob(full_text).sentiment.polarity
        day_of_week_display = pd.to_datetime(str(date)).day_name()
        is_weekend_display = int(day_of_week_display in ['Saturday', 'Sunday'])
        text_length_display = len(full_text)

        # Create input DataFrame with ONLY the 'text' column
        # Your PyCaret model's pipeline will handle the extraction of
        # HeadingLength, Sentiment, IsWeekend, text_length from this 'text' column.
        input_df = pd.DataFrame([{
            'text': full_text # Pass only the raw text to the model
        }])

        # Make prediction
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"🎯 Predicted Virality Level: {prediction}")

            # Optionally display the engineered features for user information
            st.info(f"""
                **Engineered Features (for your reference):**
                - Heading Length: {heading_length_display}
                - Sentiment: {sentiment_display:.2f}
                - Is Weekend: {'Yes' if is_weekend_display else 'No'}
                - Total Text Length: {text_length_display}
            """)

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
            st.info("Please ensure your model expects the exact features (likely just 'text') and format provided, and all necessary libraries (like PyCaret if used for training) are installed. The model's pipeline is expected to generate features like 'HeadingLength', 'Sentiment', and 'IsWeekend' internally from the 'text' column.")

# --- Additional Notes/Suggestions for Improvement ---
st.sidebar.header("💡 Suggestions for Improvement")
st.sidebar.markdown("""
1.  **Model Training Consistency:** The most common cause of "feature names unseen at fit time" is a mismatch between features provided during training and prediction. If your `best_news_classifier.pkl` was trained with PyCaret's `setup()` function and it automatically extracted features like `HeadingLength`, `Sentiment`, `IsWeekend` from a raw `text` column, then you should *only* pass the `text` column to `model.predict()`.
2.  **PyCaret Workflow:** When using PyCaret, it's common to save the entire pipeline (including preprocessing steps) using `save_model()`. When loading with `load_model()`, you typically just pass the raw data (e.g., a DataFrame with just the `text` column) and the pipeline handles all transformations.
3.  **More Features:**
    * **Readability Scores:** Incorporate Flesch-Kincaid, Gunning Fog, etc., using libraries like `textstat`.
    * **Keyword Analysis:** Extract keywords from the article and headline.
    * **Numerical Presence:** Count numbers or special characters in the headline.
    * **Time of Day:** If your data includes time, the hour of publication could be a feature.
    * **Topic Modeling:** Use techniques like LDA or NMF to identify the main topic of the article, which could influence virality.
4.  **User Experience:**
    * **Loading Spinner:** Add `st.spinner("Predicting...")` before the prediction call for better UX.
    * **Clearer Instructions:** Add a brief explanation of what each input field is for.
    * **Model Explanation:** If possible, explain *why* a prediction was made (e.g., "High sentiment and short heading contributed to high virality"). This often requires model interpretability techniques (e.g., SHAP, LIME).
5.  **Dependencies:** Make sure all dependencies (`streamlit`, `pandas`, `joblib`, `textblob`, `pycaret` (if used for model training)) are listed in a `requirements.txt` file for easy deployment.
""")
