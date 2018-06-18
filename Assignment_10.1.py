#importing libraries
import numpy as np
import pandas as pd
import string

#importing the dataset
train_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header = None, sep=',\s', na_values=["?"])
test_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', skiprows = 1, header = None, sep=',\s', na_values=["?"])
col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status','occupation','relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week','native_country', 'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels

#combining the train_set and test_set
df = pd.concat([train_set, test_set])

#Inspecting the dataset carefully
df.info()
df.isnull().sum()
df['workclass'].value_counts() # 2799 nan values with mode of PRIVATE
df['education'].value_counts()
df['marital_status'].value_counts()
df['occupation'].value_counts() #2809 nan values with mode of prof-speciality
df['relationship'].value_counts()
df['race'].value_counts()
df['sex'].value_counts()
df['native_country'].value_counts() # 857 nan values with mode of United-stayes
df['wage_class'].value_counts()

#Replacing the nan values with Modes.

#Education
df['workclass'].value_counts(sort=True)
df['workclass'].fillna('Private',inplace=True)


#Occupation
df['occupation'].value_counts(sort=True)
df['occupation'].fillna('Prof-specialty',inplace=True)


#Native Country
df['native_country'].value_counts(sort=True)
df['native_country'].fillna('United-States',inplace=True)


#Re_checking for the nan values
df.isnull().sum() #Zero nan values.

#Let's count the number of unique values from character variables.
cat = df.select_dtypes(include=['O'])
cat.apply(pd.Series.nunique)

#load sklearn and encode all object type variables
from sklearn import preprocessing

for x in df.columns:
    if df[x].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df[x].values))
        df[x] = lbl.transform(list(df[x].values))
        
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix


X = df.iloc[:,:-1].values
y = df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

#train the RF classifier
clf = RandomForestClassifier(n_estimators = 500, max_depth = 6)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)


