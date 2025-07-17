import streamlit as st
from model import analyze_sentiment, analyze_batch_sentiments  # Import from model.py
import pandas as pd  # If using your CSV for demos

st.title("Sentiment Analysis Web App")

# Single text input
user_input = st.text_area("Enter text for sentiment analysis:")
if st.button("Analyze"):
    if user_input:
        result = analyze_sentiment(user_input)
        st.write(f"Sentiment: **{result['label']}** (Confidence: {result['score']:.2f})")
    else:
        st.write("Please enter some text.")

# Batch processing (e.g., upload CSV or use your IMDBDataset.csv)
uploaded_file = st.file_uploader("Upload a CSV for batch analysis (column: 'review')")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'review' in df.columns:  # Adjust column name as needed
        results = analyze_batch_sentiments(df['review'].tolist())
        df['sentiment'] = [res['label'] for res in results]
        df['confidence'] = [res['score'] for res in results]
        st.dataframe(df)  # Display results
        # Optional: Visualize
        st.bar_chart(df['sentiment'].value_counts())
    else:
        st.write("CSV must have a 'review' column.")

# Optional: Demo with your IMDBDataset.csv
if st.button("Test with IMDB Sample"):
    imdb_df = pd.read_csv("IMDBDataset.csv")  # Load your dataset
    sample_texts = imdb_df['review'].head(5).tolist()  # Assume column 'review'
    results = analyze_batch_sentiments(sample_texts)
    for text, res in zip(sample_texts, results):
        st.write(f"Text: {text[:100]}... → {res['label']} ({res['score']:.2f})")