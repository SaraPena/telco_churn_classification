import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from scipy import stats

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# ignore_warnings
import warnings
warnings.filterwarnings("ignore")

import acquire
import prepare

train = prepare.train
test = prepare.test

train_ohe = prepare.train_ohe
test_ohe = prepare.test_ohe

train_ohe.info()
# We need to change our target variable of 'churn' into a numberic value.

# To create churn as a numberic column of 0's and 1's create a list comprehension to run the values of the column through.
train_ohe.churn.value_counts()

train_ohe['churn'] = [0 if i == 'No' else 1 for i in train_ohe.churn]

# Check the value counts
train_ohe.churn.value_counts()

# Create the train datasets:

# Create the train datasets
X_train = train_ohe[['Electronic check',
                      'Mailed check',
                      'Bank transfer (automatic)',
                      'Credit card (automatic)',
                      'DSL',
                      'Fiber optic',
                      'None',
                      'Month-to-month',
                      'One year',
                      'Two year',
                      'monthly_charges',
                      'tenure',
                      'total_charges']]

X_train2 = train_ohe[['payment_type_id',
                      'contract_type_id',
                      'internet_service_type_id',
                      'senior_citizen',
                      'tenure',
                      'monthly_charges',
                      'total_charges',
                      'tenure_years']]

y_train = train_ohe[['churn']]


# Look at our X_train.head()
X_train.head()

# Check our target variable values.
y_train.churn.value_counts()

# Logistic regression model:

# Create Logistic Regression model
logit = LogisticRegression(C=1, class_weight={1:2}, random_state = 123, solver='saga')
logit2 = LogisticRegression(C=1, class_weight={1:2}, random_state = 123, solver='saga')

# fit model to train data
logit.fit(X_train,y_train)
logit2.fit(X_train2,y_train)

# Create predicts:
y_pred = logit.predict(X_train)
y_pred2 = logit2.predict(X_train2)

# look at accuracy of models:
accuracy = logit.score(X_train,y_train)
accuracy2 = logit2.score(X_train2,y_train)
accuracy, accuracy2

# Create confusion matrix.
confusion_matrix(y_train, y_pred)

confusion_matrix(y_train,y_pred2)

# Create classification report:
print(classification_report(y_train,y_pred2))

# Decision tree model:

# Create model object:
clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, random_state = 123)
clf2 = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, random_state = 123)

# fit the model:

clf.fit(X_train, y_train)
clf2.fit(X_train2, y_train)

# Create predictions:
y_pred_dt = clf.predict(X_train)
y_pred_dt2 = clf2.predict(X_train2)

# Look at accuracy:
clf.score(X_train,y_train)
clf2.score(X_train2,y_train)

# Create confusion matrix:
confusion_matrix(y_train,y_pred_dt)
confusion_matrix(y_train,y_pred_dt2)

# Create classification report
print(classification_report(y_train, y_pred_dt))

# Random Forest:

# Create model object:
rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='entropy',
                            min_samples_leaf=5,
                            n_estimators=100,
                            max_depth=3, 
                            random_state=123)

rf2 = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='entropy',
                            min_samples_leaf=5,
                            n_estimators=100,
                            max_depth=3, 
                            random_state=123)

# Fit the model to train data:
rf.fit(X_train,y_train)
rf2.fit(X_train2,y_train)

# Look at feature importances:
rf.feature_importances_
rf2.feature_importances_

# Create predictions:
y_pred_rf = rf.predict(X_train)
y_pred_rf2 = rf2.predict(X_train2)

# Look at accuracy:
rf.score(X_train, y_train), rf2.score(X_train2,y_train)

# Create confusion matrix:
confusion_matrix(y_train, y_pred_rf)
confusion_matrix(y_train,y_pred_rf2)

# Print classification report:
print(classification_report(y_train,y_pred_rf2))

# KNN
# Want to avoid too many dimensions will only select based on internet service fiber optic, contract type, tenure.
X_train_mod = train_ohe[['Fiber optic',
                      'Month-to-month',
                      'tenure',
                      'total_charges']]

# Create KNN object
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn2 = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn_mod = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# Fit model to the train data:
knn.fit(X_train,y_train)
knn2.fit(X_train2,y_train)
knn_mod.fit(X_train_mod,y_train)

# Create prediction values:
y_pred_knn = knn.predict(X_train)
y_pred_knn2 = knn2.predict(X_train2)
y_pred_knn_mod = knn_mod.predict(X_train_mod)


# Look at accuracy:
knn.score(X_train,y_train), knn2.score(X_train2,y_train), knn_mod.score(X_train_mod,y_train)

# Create confusion matricies:
confusion_matrix(y_train, y_pred_knn)
confusion_matrix(y_train, y_pred_knn2)

print(classification_report(y_train,y_pred_knn))

# Simulation models:
# If we predict no one to churn:
y_sim = [0 for i in range(0,len(y_train.churn))]

# Classification report:
print(classification_report(y_train,y_sim))

# Selecting a random number between 0 and 1 for 4922 trials (same as test data)
a = np.random.random((4922,1))
a[0:5]

# Establish the train population churn_rate
churn_rate = len(train.churn[train.churn == 'Yes'])/len(train.churn)

# Where number is less than churn rate we will call that a "success" (the customer churned)
t = (a < churn_rate).sum(axis = 1)
t

# Classification report:
print(classification_report(y_train, t))


# Test Data
test_ohe['churn'] = [0 if i == 'No' else 1 for i in test_ohe.churn]

# Check value counts:
test_ohe.churn.value_counts()

X_test = test_ohe[['Electronic check',
                      'Mailed check',
                      'Bank transfer (automatic)',
                      'Credit card (automatic)',
                      'DSL',
                      'Fiber optic',
                      'None',
                      'Month-to-month',
                      'One year',
                      'Two year',
                      'monthly_charges',
                      'tenure',
                      'total_charges']]

y_test = test_ohe[['churn']]

# Use logistic regression: had the highest recall rate.
# Create & fit the model:
logit = LogisticRegression(C=1, class_weight={1:2}, random_state = 123, solver='saga')

logit.fit(X_test,y_test)

# Make predictions:
y_pred_test = logit.predict(X_test)

# Score the model:
logit.score(X_test,y_test)

print(classification_report(y_test,y_pred_test))

# These recall scores are similar to our recall scores with the train data set

# Create our probability predictions for the entire df.

st = prepare.ohe_df
# st.info()

st['churn'] = [0 if i == 'No' else 1 for i in st.churn]

y = st[['churn']]
X = st[['Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)',
        'DSL',
        'Fiber optic',
        'None',
        'Month-to-month',
        'One year',
        'Two year',
        'monthly_charges',
        'tenure',
        'total_charges']]

logit = LogisticRegression(C=1, class_weight={1:2}, random_state = 123, solver='saga')

logit.fit(X,y)

y_pred = pd.DataFrame(logit.predict(X), columns = ['y_pred'],index = st.index)

print(classification_report(y,y_pred))

# Predict the probabilities of churn using logistic regression model:
y_pred_proba = pd.DataFrame(logit.predict_proba(X), columns = ['no_churn_proba','churn_proba'], index = st.index)


# Create probability dataframe
proba_df = y.join([y_pred,y_pred_proba])

# export this file to a CSV
#export_csv = proba_df.to_csv (r'proba_churn.csv', header=True) #Don't forget to add '.csv' at the end of the path
#export_csv = df.to_csv (r'C:\Users\Ron\Desktop\export_dataframe.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path



