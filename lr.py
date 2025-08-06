# digit-recognizer from kaggle and using logistic regression rn, will try KNN later

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#data
print("Loading data...")
train_data = pd.read_csv("train.csv") 
print("Data loaded ")

print(train_data.shape) 

# seperating features
X = train_data.drop("label", axis=1)  #so that the model doesnt read this
y = train_data["label"]              

#splitting data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=28)
print("Data split")

# using logistic reg to train
model = LogisticRegression(max_iter=700) 
model.fit(X_train, y_train)
print("Model training complete!")

#  predictions
print("Making predictions now:")
y_pred = model.predict(X_test)

# to check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of my model: {accuracy * 100:.2f}%")

#uh the accuracy didnt turn out great, so either ill just push it to my repo and then work on it later or ill just push and maybe do the knn approach mhm