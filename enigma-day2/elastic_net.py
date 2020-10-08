import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
df = pd.read_csv("train.csv")

#print(df.dtypes) #function giving the datatypes of all columns
features = ['People','Population','Width(m)','Length(m)','Months']


X = df[features]
Y = df['Survival ']

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






def data_normalizer(train_data, type_normalization):
  if type_normalization == 'mean':
    train_data_n = (train_data-train_data.mean())/train_data.std()
    #test_data_n = (test_data-test_data.mean())/test_data.std()
  elif type_normalization =='minmax':
    train_data_n = (train_data-train_data.min())/(train_data.max()-train_data.min())
    #test_data_n = (test_data-test_data.min())/(test_data.max()-test_data.min())

  return train_data_n


X = data_normalizer(X,'mean')
Y = data_normalizer(Y,'mean')





from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25)

from sklearn.linear_model import ElasticNet

regr = ElasticNet(random_state=42,normalize = True,alpha = 0)
regr.fit(x_train,y_train)

y_pred = regr.predict(x_test)

print("rms test  error:",mean_squared_error(y_test, y_pred,squared = False))



df_train = pd.read_csv("test.csv")


df.replace(people_replace_dict,inplace = True)

df_train_pred = df_train[features]



predictions = regr.predict(df_train_pred)

pred=pd.DataFrame(predictions)
sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','Survival %']
datasets.to_csv('my_submission.csv',index=False)
