# --------------
# Importing libraries
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

#Code Starts here
# Load the train data stored in path variable
data_train=pd.read_csv(path)

# Load the test data stored in path1 variable
data_test=pd.read_csv(path1)

# Remove rows with incorrect labels in test dataset
data_test.dropna(subset=['Target'],inplace=True)
print(data_test.head(),data_train.head())

# Encode target variable as integer
target_train={'<=50K':0,'>50K':1}
target_test={'<=50K.':0,'>50K.':1}
data_train.Target.replace(target_train,inplace=True,regex=True)
data_test.Target.replace(target_test,inplace=True,regex=True)
print(data_test.head(),'\n',data_train.head())

# Plot the distribution of each feature
def create_plots(data,rows=5,cols=3):

  fig=plt.figure(figsize=(20,25))

  for i,column in enumerate(data.columns):
    ax=fig.add_subplot(rows,cols,i+1)
    ax.set_title(column)
    if data[column].dtypes == np.object:
      data[column].value_counts().plot(kind='bar',axes=ax)
    else:
      data[column].hist(axes=ax)
      plt.xticks(rotation=90)
  plt.subplots_adjust(hspace=0.9,wspace=0.3)

create_plots(data_train)
create_plots(data_test)

# Convert the data type of Age column in the test data to int type
data_test.Age=data_test.Age.astype(int)

# Cast all float features to int type to keep types consistent between our train and test data

data_test.fnlwgt=data_test.fnlwgt.astype(int)         
data_test.Education_Num=data_test.Education_Num.astype(int)   
data_test.Capital_Gain=data_test.Capital_Gain.astype(int)  
data_test.Capital_Loss=data_test.Capital_Loss.astype(int)  
data_test.Hours_per_week=data_test.Hours_per_week.astype(int) 
dt= data_train.dtypes==data_test.dtypes
print(dt[dt==False])

# Choose categorical and continuous features from data and print them
categorical_columns=data_train.select_dtypes(include=object).columns.tolist()
print(categorical_columns)

numerical_columns=data_train.select_dtypes(include='number').columns.tolist()
print(numerical_columns)

# Fill missing data for catgorical columns
for c in categorical_columns:
  data_train[c].fillna(data_train[c].mode()[0],inplace=True)
  data_test[c].fillna(data_train[c].mode()[0],inplace=True)

                                       
# Fill missing data for numerical columns   
for c in numerical_columns:
  data_train[c].fillna(data_train[c].median(),inplace=True)
  data_test[c].fillna(data_train[c].median(),inplace=True)                                   

# Dummy code Categoricol features
#le=LabelEncoder()
#for x in categorical_columns:
#  data_train[x]=le.fit_transform(data_train[x])
#  data_test[x]=le.transform(data_test[x])

#One Hot Encoding 
data_train_ohe=pd.get_dummies(data=data_train,columns=categorical_columns)
data_test_ohe=pd.get_dummies(data=data_test,columns=categorical_columns)

# Check for Column which is not present in test data
list(set(data_train_ohe.columns.tolist())-set(data_test_ohe.columns.tolist()))

# New Zero valued feature in test data for Holand
data_test_ohe['Country_14']=0

# Split train and test data into X_train ,y_train,X_test and y_test data
X_train=data_train_ohe.drop(['Target'],axis=1)
y_train=data_train_ohe['Target']

X_test=data_test_ohe.drop(['Target'],axis=1)
y_test=data_test_ohe['Target']

# Train a decision tree model then predict our test data and compute the accuracy
tree=DecisionTreeClassifier(max_depth=3,random_state=17)
tree.fit(X_train,y_train)
tree_pred=tree.predict(X_test)
print('accuracy',accuracy_score(y_test,tree_pred))

# Decision tree with parameter tuning
tree_params={'max_depth':range(2,11),'min_samples_leaf':range(10,100,10)}
tuned_tree=GridSearchCV(DecisionTreeClassifier(random_state=17),tree_params,cv=5)
tuned_tree.fit(X_train,y_train)

# Print out optimal maximum depth(i.e. best_params_ attribute of GridSearchCV) and best_score_
print(tuned_tree.best_params_)
print(tuned_tree.best_score_)

#Train a decision tree model with best parameter then predict our test data and compute the accuracy
final_tuned_tree=DecisionTreeClassifier(max_depth=10,min_samples_leaf=10,random_state=17)
final_tuned_tree.fit(X_train,y_train)
y_pred=final_tuned_tree.predict(X_test)
print('final_tuned_tree accuracy',accuracy_score(y_test,y_pred))

from sklearn.metrics import classification_report,confusion_matrix
print('confusion matrix',confusion_matrix(y_test,y_pred))
print('classification report',classification_report(y_test,y_pred))

#Code ends here


