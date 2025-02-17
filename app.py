import streamlit as st
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# Load API Key from Streamlit Secrets
FMP_API_KEY = st.secrets["FMP_API_KEY"]

# Load FinBERT Model
@st.cache_resource
def load_finbert():
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_finbert()

# Function to fetch the latest earnings transcript
def fetch_earnings_transcript(symbol):
    url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}?apikey={FMP_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        transcripts = response.json()
        if transcripts:
            return transcripts[0]["content"]  # Extract latest transcript
    return None

# Function to analyze sentiment with FinBERT
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    labels = ["Negative", "Neutral", "Positive"]
    return {labels[i]: scores[0][i].item() for i in range(len(labels))}

# Streamlit UI
st.title("📈 Earnings Transcript Sentiment Analyzer")

symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):").upper()

if st.button("Analyze Transcript"):
    if symbol:
        st.write(f"Fetching latest earnings transcript for **{symbol}**...")
        transcript = fetch_earnings_transcript(symbol)

        if transcript:
            st.success("Transcript successfully fetched!")
            st.text_area("Earnings Transcript (First 1000 chars shown)", transcript[:1000], height=200)

            # Sentiment Analysis
            st.write("🔍 **Analyzing sentiment...**")
            sentiment_scores = analyze_sentiment(transcript)

            st.write("### Sentiment Breakdown")
            st.json(sentiment_scores)

            # Pie Chart Visualization
            import matplotlib.pyplot as plt

            labels = sentiment_scores.keys()
            sizes = sentiment_scores.values()
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=["red", "gray", "green"])
            ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)
        else:
            st.error("No transcript found for this ticker. Try another symbol.")
    else:
        st.warning("Please enter a stock ticker.")
