import streamlit as st
import joblib
import re

# Load the pre-trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Web UI - Title and Description with improved styling
st.markdown("<h1 style='text-align: center; color: #1E88E5;'>📰 Fake News Detection System</h1>", unsafe_allow_html=True)
st.write("Enter the news article text to check if it's *Real* or *Fake*.")

# Text input from the user
user_input = st.text_area("Enter News Text:")

# Function to check if the input text is valid
def is_valid_input(text):
    # Remove any non-alphabetic characters and check if there's enough meaningful content
    text = text.strip()
    if len(text) == 0:
        return False  # Empty input is not valid
    if len(re.findall(r'\b\w+\b', text)) < 2:  # Check if there are at least two words in the text
        return False  # Invalid if less than 2 words (this can be adjusted based on your needs)
    return True  # Valid if the text has meaningful content

# Button to trigger prediction
if st.button("Predict"):
    if user_input:
        # Validate the input text
        if not is_valid_input(user_input):
            st.warning("Please enter a valid news text! The input seems too random or meaningless.")
        else:
            # Convert the text input to a vector (numerical representation)
            user_input_vec = vectorizer.transform([user_input])

            # Make prediction
            prediction = model.predict(user_input_vec)
            prediction_prob = model.predict_proba(user_input_vec)

            # Show the result with confidence
            if prediction[0] == 1:
                st.success(f"✅ *Real News* with {prediction_prob[0][1]*100:.2f}% confidence.")
            else:
                st.error(f"❌ *Fake News* with {prediction_prob[0][0]*100:.2f}% confidence.")
    else:
        st.warning("Please enter some text!")

# Option to input multiple pieces of news for prediction
st.write("---")
st.write("### Test Multiple News Articles")
multiple_news = st.text_area("Enter multiple news articles (separate each by a newline):")

if st.button("Predict All"):
    if multiple_news:
        news_list = multiple_news.split('\n')
        for news in news_list:
            news = news.strip()  # Remove extra spaces
            if news:
                # Validate the input text
                if not is_valid_input(news):
                    st.warning(f"News text: '{news}' seems too random or meaningless. Please enter valid text.")
                else:
                    user_input_vec = vectorizer.transform([news])
                    prediction = model.predict(user_input_vec)
                    prediction_prob = model.predict_proba(user_input_vec)
                    st.write(f"*News*: {news}")
                    if prediction[0] == 1:
                        st.success(f"✅ *Real News* with {prediction_prob[0][1]*100:.2f}% confidence.")
                    else:
                        st.error(f"❌ *Fake News* with {prediction_prob[0][0]*100:.2f}% confidence.")
            else:
                st.warning("Please enter valid news text!")
    else:
        st.warning("Please enter some news articles to predict.")