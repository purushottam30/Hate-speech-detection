from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import nltk
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle

# Initialize Flask app
app = Flask(__name__)

# Download NLTK stopwords
nltk.download('stopwords')
file_path = 'D:/Hate speech detection/twitter.csv'
# Read the CSV file
data = pd.read_csv(file_path)

# Map class labels to more descriptive names
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "Normal"})
data = data[["tweet", "labels"]]

# Initialize the stemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Clean the data
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = " ".join(text.split())  # Ensure single spaces
    text = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(text)

data["tweet"] = data["tweet"].apply(clean)

# Check for any empty documents after cleaning
data = data[data["tweet"].str.strip().astype(bool)]

# Split data into features and labels
x = np.array(data["tweet"])
y = np.array(data["labels"])

# Vectorize the text data
cv = CountVectorizer()
X = cv.fit_transform(x)

# Balance the dataset using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Train the models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

metrics = {}
for model_name, model in models.items():
    model.fit(X_resampled, y_resampled)
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, average='weighted')
    recall = recall_score(y, predictions, average='weighted')
    f1 = f1_score(y, predictions, average='weighted')
    metrics[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    with open(f'{model_name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)

# Load the models and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

loaded_models = {}
for model_name in models.keys():
    with open(f'{model_name.lower().replace(" ", "_")}_model.pkl', 'rb') as f:
        loaded_models[model_name] = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        predictions = {name: model.predict(vect)[0] for name, model in loaded_models.items()}
        return render_template('index.html', prediction=predictions)

@app.route('/metrics')
def metrics_page():
    return render_template('metrics.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)
