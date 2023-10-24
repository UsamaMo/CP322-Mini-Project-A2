from sklearn.datasets import fetch_20newsgroups
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

#Generate Random Data
X, y = make_classification(n_samples=1000, n_features=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
    # Train
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc}")






#--------------------------------------------------------------------------------------------------------------------
#JUST AN EXAMPLE FOR NEWSGROUP
#--------------------------------------------------------------------------------------------------------------------

#this code was given by the newsgroup tutorial website posted in the assignment instructions
#https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html


categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

#just testing to see if the function works,  takes too long to fetch data
print('start fetching')
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
print('stop fetching')


#prints Target names
print(twenty_train.target_names)               

#prints the length of data and filename
print(len(twenty_train.data))                  
print(len(twenty_train.filenames))


#prints the first 3 lines of the first data file
print("\n".join(twenty_train.data[0].split("\n")[:3]))

#print target name of the first data file
print(twenty_train.target_names[twenty_train.target[0]])

#print first 10 targets
print(twenty_train.target[:10])

#print target name of the first 10 data files
for t in twenty_train.target[:10]:
        print(twenty_train.target_names[t])



#--------------------------------------------------------------------------------------------------------------------
#BEGIN coding below, above are just examples 
#--------------------------------------------------------------------------------------------------------------------



