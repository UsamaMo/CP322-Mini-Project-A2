import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split


# Function to read data
# If use_subset is True, then it only read a dataset, if its false it reads the full dataset
# This use subset function was added so that testing was easier to do faster with this large dataset
def read_data(path, use_subset=False, max_per_category=100):

    texts, labels = [], []
    for label in ['pos', 'neg']:
        dir_path = os.path.join(path, label)
        files = os.listdir(dir_path)
        # If use_subset is True, limit the number of files read
        if use_subset:
            files = files[:max_per_category]
        for file in files:
            file_path = os.path.join(dir_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1 if label == 'pos' else 0)
    return texts, labels

# Specify the path to your dataset
train_path = 'aclImdb/train'
test_path = 'aclImdb/test'

# Use the modified read_data function to read a subset of the dataset
#To read the full dataset also change use_subset to False so that it read the full dataset
train_texts, train_labels = read_data(train_path, use_subset=False, max_per_category=10)
test_texts, test_labels = read_data(test_path, use_subset=False, max_per_category=10)


# Reading Data
train_texts, train_labels = read_data('aclImdb/train')
test_texts, test_labels = read_data('aclImdb/test')

# Vectorize texts with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Create and train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": LinearSVC(dual=False,  max_iter=1000),
    "AdaBoost": AdaBoostClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train, predict and evaluate models
for name, model in models.items():
    model.fit(X_train, train_labels)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred)
    recall = recall_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred)
    print(f"{name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")