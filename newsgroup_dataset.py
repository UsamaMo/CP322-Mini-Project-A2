from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import matplotlib.pyplot as plt

# Load the 20 newsgroups dataset
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)
y_train = newsgroups_train.target
y_test = newsgroups_test.target

# Define the 5 different models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": LinearSVC(dual=False),
    "AdaBoost": AdaBoostClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Initialize dictionaries to store metrics
accuracy_scores = {}
precision_scores = {}

# Train, predict, and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate each metric with the 'weighted' average for multi-class classification
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')

    # Print the metrics
    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")

    # Store each metric in dictionaries
    accuracy_scores[name] = accuracy
    precision_scores[name] = precision

# Function to create bar plots for each metric
def plot_metrics(metric_scores, title):
    plt.figure(figsize=(10, 6))  
    plt.bar(metric_scores.keys(), metric_scores.values())
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title(title)
    plt.show()

# Plot the Accuracy and the Precision
plot_metrics(accuracy_scores, 'Newsgroup Model Accuracy')
plot_metrics(precision_scores, 'Newsgroup Model Precision')
