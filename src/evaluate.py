import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import load_and_process
from sklearn.model_selection import train_test_split

# Load model and vectorizer
with open("xgb_model.pkl", "rb") as f:
    xgb = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Load and preprocess data
df = load_and_process("data/Fake.csv", "data/True.csv")
X = df["clean"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Transform
X_test_vec = tfidf.transform(X_test)

# Predict
y_pred = xgb.predict(X_test_vec)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
