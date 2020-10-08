import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv("train.csv")
#print(df.dtypes) #function giving the datatypes of all columns
features = ['People','Population','Width(m)','Length(m)','Months']

people_replace_dict = {
  "People" :
	{"Slavs" : 0,
	"Poles" : 1,
	"POWs" :2,
	"Parkki" :3,
	"Jews" :4,
	"Undesirables" :5,
	"Handicapped" :6
	}
}

df.replace(people_replace_dict,inplace = True)

#print(df.tail())

X = df[features]
Y = df['Survival ']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25)

print('X shape:'+str(X.shape))
print('Y shape:'+str(Y.shape))
print('X_train shape:'+str(x_train.shape))
print('X_test shape:'+str(x_test.shape))
print('y_train shape:'+str(y_train.shape))
print('y_test shape:'+str(y_test.shape))

#corrmat = X.corr()
#f, ax = plt.subplots(figsize=(20, 9))
#sns.heatmap(corrmat, vmax=.8, annot=True);
#plt.show()


#sns.set()
#sns.pairplot(X, size = 2.5)
#plt.show();
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet


regressor = ElasticNet(fit_intercept = True,normalize = True,precompute = False)



hyperparameter_grid = {
  'max_iter' :[50,100,150,200]
    }



random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_root_mean_squared_error',n_jobs = -1,
            verbose = 5, 
            return_train_score = True,
            random_state=42)

random_cv.fit(x_train,y_train)

print(random_cv.best_estimator_)

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
y_pred_train = regressor.predict(x_train)
print("rms train  error:",mean_squared_error(y_train, y_pred_train,squared = False))
print("rms test  error:",mean_squared_error(y_test, y_pred,squared = False))


df_train = pd.read_csv("test.csv")

df_train.replace(people_replace_dict,inplace = True)


df_train_pred = df_train[features]




predictions = regressor.predict(df_train_pred)

pred=pd.DataFrame(predictions)
sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','Survival %']
datasets.to_csv('my_submission.csv',index=False)






