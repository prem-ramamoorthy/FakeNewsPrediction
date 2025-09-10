import pickle
from preprocess import clean_text

# Load model and vectorizer
with open("xgb_model.pkl", "rb") as f:
    xgb = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

def predict_news(news_text):
    clean = clean_text(news_text)
    vec = tfidf.transform([clean])
    pred = xgb.predict(vec)
    return "Real" if pred[0] == 1 else "Fake"

# Example usage
if __name__ == "__main__":
    sample = "Breaking news: Scientists discover a new planet in our solar system."
    result = predict_news(sample)
    print(f"Prediction: {result}")
