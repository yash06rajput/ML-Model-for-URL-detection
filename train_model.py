import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("url_dataset.csv")

# Clean the data
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Encode string labels to numeric
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['url'])
y = df['label_encoded']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model, vectorizer, and label encoder
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model trained with string labels and saved.")
