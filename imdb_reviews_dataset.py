import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


#--------------------------------------------------------------------------------------------------------------------
#JUST A PLACEHOLDER, CHANGE THE CODE BELOW IF NEEDED
#--------------------------------------------------------------------------------------------------------------------



# Function to read data
def read_data(path):
    texts, labels = [], []
    for label in ['pos', 'neg']:
        dir_path = os.path.join(path, label)
        for file in os.listdir(dir_path):
            with open(os.path.join(dir_path, file), 'r', encoding='utf-8') as f:

                texts.append(f.read())
                labels.append(1 if label == 'pos' else 0)
    return texts, labels

# Reading Data
train_texts, train_labels = read_data('aclImdb/train')
test_texts, test_labels = read_data('aclImdb/test')

# Vectorize texts
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

#Create Random models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": LinearSVC(),
    "AdaBoost": AdaBoostClassifier(),
    "Random Forest": RandomForestClassifier()
}

#Train, predict and evaluate random models
for name, model in models.items():
    model.fit(X_train, train_labels)
    y_pred = model.predict(X_test)
    acc = accuracy_score(test_labels, y_pred)
    print(f"{name} Accuracy: {acc}")

