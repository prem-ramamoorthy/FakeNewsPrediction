import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords")

ps = PorterStemmer()
stop_words = stopwords.words("english")
stop_words.extend(["from", "subject", "re", "edu", "use"])

def clean_text(text):
    """Clean text: remove HTML, numbers, punctuation, lowercase, remove stopwords, stem."""
    if not text:
        text = ""
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove non-alphabetic characters
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    filtered_words = [ps.stem(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(filtered_words)

def load_and_process(fake_path, true_path):
    # Load datasets
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # Keep only title + text
    df_fake = df_fake.iloc[:, 0:2]
    df_true = df_true.iloc[:, 0:2]

    # Add label
    df_fake["label"] = 0
    df_true["label"] = 1

    # Combine
    df = pd.concat([df_fake, df_true], ignore_index=True)
    df["title_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["clean"] = df["title_text"].apply(clean_text)

    return df[["clean", "label"]]