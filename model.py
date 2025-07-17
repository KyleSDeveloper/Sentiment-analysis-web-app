from transformers import pipeline

# Load the pre-trained sentiment analysis pipeline
# Option 1: Lightweight DistilBERT (faster, good for demos)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Option 2: Uncomment for higher accuracy with RoBERTa (slower, but better generalization)
# sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

def analyze_sentiment(text):
    """
    Analyze sentiment of input text.
    Returns: {'label': 'POSITIVE' or 'NEGATIVE', 'score': confidence (0-1)}
    """
    result = sentiment_pipeline(text)[0]  # Pipeline returns a list; take first item
    return result

def analyze_batch_sentiments(texts):
    """
    Analyze a list of texts (for batch processing).
    Returns list of results with original text attached.
    """
    clean_texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    results = sentiment_pipeline(clean_texts)
    return [{"text": t, **r} for t, r in zip(clean_texts, results)]
