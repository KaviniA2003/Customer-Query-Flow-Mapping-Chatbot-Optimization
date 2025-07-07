# scripts/train_model.py

import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def main():
    # Load data
    from google.colab import files
    uploaded = files.upload()

    df = pd.read_csv(list(uploaded.keys())[0], on_bad_lines='skip')

    # Clean data
    df = df.dropna(subset=["user_msg", "intent"])
    df = df[df["intent"].str.lower() != "unknown"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["user_msg"], df["intent"], test_size=0.2, random_state=42
    )

    # Build pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Train model
    print("🚀 Training intent classification model...")
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("\n📊 Evaluation Results:")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs("updated_model", exist_ok=True)
    model_path = os.path.join("updated_model", "intent_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\n✅ Model saved to: {model_path}")

if __name__ == "__main__":
    main()
