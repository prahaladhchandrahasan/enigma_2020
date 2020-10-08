import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Graphs for DataAnalysis

def correlation(train_data):
  train_corr = train_data.corr()
  sns.heatmap(train_corr, annot=True)
  plt.show()

def PairPlot(train_data,hue):
	sns.pairplot(train_data,hue)
	plt.show()
def bar_chart(feature,Y,train_data):
    won = train_data[Y==1][feature].value_counts()
    lost = train_data[Y==2][feature].value_counts()
    df = pd.DataFrame([won,lost])
    df.index = ['Won','Lost']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
    plt.show()


# Normalization  and Logarithmictransformation


def data_normalizer(train_data, type_normalization):
  if type_normalization == 'mean':
    train_data_n = (train_data-train_data.mean())/train_data.std()
    #test_data_n = (test_data-test_data.mean())/test_data.std()
  elif type_normalization =='minmax':
    train_data_n = (train_data-train_data.min())/(train_data.max()-train_data.min())
    #test_data_n = (test_data-test_data.min())/(test_data.max()-test_data.min())

  return train_data_n


def logarithic_transformation(train_data) :
	t_log = (train_data+1).trasform(np.log)
	return t_log


# Train Test split

def splits(X,Y,test_split_ratio):

	X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size=test_split_ratio, random_state=42)
	return X_train,X_test,y_train,y_test










