import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels: 0 for Fake, 1 for Real
fake["label"] = 0
true["label"] = 1

# Combine both into one dataset
data = pd.concat([fake, true])
data = data[['title', 'text', 'label']]  # Use only needed columns

# Shuffle the rows
data = data.sample(frac=1).reset_index(drop=True)

# Combine title and text into one column
data["content"] = data["title"] + " " + data["text"]

# Define input (X) and output (y)
X = data["content"]
y = data["label"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Test the model
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Save the model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# -------------------------
# Predict on a new sample:
sample_news = "The government has announced new subsidies for farmers in the 2025 budget."

sample_vec = vectorizer.transform([sample_news])
sample_pred = model.predict(sample_vec)

if sample_pred[0] == 1:
    print("Prediction: ✅ Real News")
else:
    print("Prediction: ❌ Fake News")