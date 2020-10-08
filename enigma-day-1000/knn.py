import numpy as np 
import pandas as pd 

df = pd.read_csv("train.csv")



features = ['Number of soldiers',
'Number of tanks',
'Number of aircrafts'
]

X = df[features]
Y = df['Victory Status']


def data_normalizer(train_data, type_normalization):
  if type_normalization == 'mean':
    train_data_n = (train_data-train_data.mean())/train_data.std()
    #test_data_n = (test_data-test_data.mean())/test_data.std()
  elif type_normalization =='minmax':
    train_data_n = (train_data-train_data.min())/(train_data.max()-train_data.min())
    #test_data_n = (test_data-test_data.min())/(test_data.max()-test_data.min())

  return train_data_n


X = data_normalizer(X,'mean')


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25)


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV



leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)#Create new KNN object
knn_2 = KNeighborsClassifier()#Use GridSearch
clf = GridSearchCV(knn_2, hyperparameters, cv=10)


clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

df_test = pd.read_csv("test.csv")
df_test_pred = df_test[features]
df_test_pred = data_normalizer(df_test_pred,'mean')

predictions = clf.predict(df_test_pred)


pred=pd.DataFrame(predictions)
sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','Category']
datasets.to_csv('my_submission.csv',index=False)