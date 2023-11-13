import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score


# Function to read data
# If use_subset is True, then it only read a dataset, if its false it reads the full dataset
# This use subset function was added so that testing was easier to do faster with this large dataset
def read_data(path, use_subset=True, max_per_category=100):

    texts, labels = [], []
    for label in ['pos', 'neg']:
        dir_path = os.path.join(path, label)
        files = os.listdir(dir_path)
        # If use_subset is True, use a subset from the dataset and not the full dataset
        if use_subset:
            files = files[:max_per_category]
        for file in files:
            file_path = os.path.join(dir_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1 if label == 'pos' else 0)
    return texts, labels

#path to test and train folders
train_path = 'aclImdb/train'
test_path = 'aclImdb/test'

# Use the modified read_data function to read a subset of the dataset
#To read the full dataset also change use_subset to False so that it read the full dataset
train_texts, train_labels = read_data(train_path, use_subset=True, max_per_category=10)
test_texts, test_labels = read_data(test_path, use_subset=True, max_per_category=10)


# Reading Data from the test and train folders
train_texts, train_labels = read_data('aclImdb/train')
test_texts, test_labels = read_data('aclImdb/test')

# Vectorize texts with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Define the 5 different models
models = {
    "Logistic Reg": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": LinearSVC(dual=False),
    "AdaBoost": AdaBoostClassifier(),
    "Random Forest": RandomForestClassifier()
}


# Initialize dictionaries to store metrics
accuracy_scores = {}
precision_scores = {}

# Function to create bar plots for each metric
def plot_metrics(metric_scores, title):
    plt.figure(figsize=(10, 6))  
    plt.bar(metric_scores.keys(), metric_scores.values())
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks()  
    plt.show()

# Train, predict, and evaluate models
for name, model in models.items():
    model.fit(X_train, train_labels)
    y_pred = model.predict(X_test)

    # Calculate each metric
    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, average='binary')

    # Store each metric in dictionaries
    accuracy_scores[name] = accuracy
    precision_scores[name] = precision

    # Print the metrics
    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")

# Plot the Accuracy and the Precision
plot_metrics(accuracy_scores, 'IMDB Reviews Model Accuracy')
plot_metrics(precision_scores, 'IMDB Reviews Model Precision')