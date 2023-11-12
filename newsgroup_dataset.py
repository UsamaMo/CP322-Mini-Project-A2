from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

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

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": LinearSVC(dual=False),
    "AdaBoost": AdaBoostClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train, predict, and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")




#--------------------------------------------------------------------------------------------------------------------
#THE CODE BELOW WAS GIVEN BY THE DATASET ITSELF TO USE FOR IMPLEMENTATION IF NEEDED
# #--------------------------------------------------------------------------------------------------------------------

# #this code was given by the newsgroup tutorial website posted in the assignment instructions
# #https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html


# categories = ['alt.atheism', 'soc.religion.christian',
#               'comp.graphics', 'sci.med']

# #just testing to see if the function works,  takes too long to fetch data
# print('start fetching')
# twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
# print('stop fetching')


# #prints Target names
# print(twenty_train.target_names)               

# #prints the length of data and filename
# print(len(twenty_train.data))                  
# print(len(twenty_train.filenames))


# #prints the first 3 lines of the first data file
# print("\n".join(twenty_train.data[0].split("\n")[:3]))

# #print target name of the first data file
# print(twenty_train.target_names[twenty_train.target[0]])

# #print first 10 targets
# print(twenty_train.target[:10])

# #print target name of the first 10 data files
# for t in twenty_train.target[:10]:
#         print(twenty_train.target_names[t])


