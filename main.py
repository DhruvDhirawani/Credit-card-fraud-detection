import numpy as pd
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


df =pd.read_csv("G:\resume Data science\Credit-card-fraud-detection\creditcard.csv.zip")
df.describe()

df.isnull().sum()

# distribution of legit transactions & fraudulent transactions
df['Class'].value_counts()

# separating the data for analysis
not_fraud = df[df.Class == 0]
fraud = df[df.Class == 1]

print(not_fraud.shape)
print(fraud.shape)

not_fraud.Amount.describe()

fraud.Amount.describe()

# compare the values for both transactions
df.groupby('Class').mean()

not_fraud_sample = not_fraud.sample(n=492) #takes a random sample of 492 rows from the not_fraud DataFrame.
new_dataset = pd.concat([not_fraud_sample, fraud], axis=0)

new_dataset.head()
new_dataset.tail()

new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()

#Splitting the data into Features & Targets
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

print(X)

print(Y)

# Import train_test_split
from sklearn.model_selection import train_test_split 


# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

len(X_train)

len(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)
