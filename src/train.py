import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from preprocess import load_and_process

# Load and preprocess data
df = load_and_process("data/Fake.csv", "data/True.csv")
X = df["clean"].values
y = df["label"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize text
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Train XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(X_train_vec, y_train)

# Save model and vectorizer
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Training completed and model saved.")
