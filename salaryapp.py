





#Employee salary prediction using adult csv
#loading the libraries
import pandas as pd
import numpy as np


salary_data=pd.read_csv('/content/adult 3.csv')


salary_data

salary_data.head()
salary_data.head()
salary_data.head()

salary_data.shape

salary_data.head(6)

salary_data.tail()

salary_data.isnull().sum()

salary_data.isna()

salary_data.isna().sum()

salary_data.info()

salary_data.value_counts('occupation')

salary_data.gender.value_counts()





salary_data.education.value_counts()

salary_data['marital-status'].value_counts()


salary_data.workclass.value_counts()

salary_data.occupation.replace({'?':'others'},inplace=True)


salary_data.occupation.value_counts()

salary_data.workclass.replace({'?':'NotListed'},inplace=True)


salary_data.workclass.value_counts()

delete the categories from columns

salary_data=salary_data[salary_data['workclass'] !='Without-pay']
salary_data=salary_data[salary_data['workclass'] !='Never-worked']

salary_data['workclass'].value_counts()

salary_data.shape

salary_data=salary_data[salary_data['education'] !='5th-6th']
salary_data=salary_data[salary_data['education'] !='1st-4th']
salary_data=salary_data[salary_data['education'] !='Preschool']

salary_data.education.value_counts()

salary_data.shape

Redundancy

salary_data=salary_data.drop(columns='education',axis=1)

#Outlier
import matplotlib.pyplot as plt
plt.boxplot(salary_data['age'])
plt.show()

salary_data=salary_data[(salary_data['age']<=75) & (salary_data['age']>=17)]

#Outlier
import matplotlib.pyplot as plt
plt.boxplot(salary_data['age'])
plt.show()

X=salary_data.drop(columns='income',axis=1)
Y=salary_data['income']

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


encoder.fit_transform(salary_data['workclass'])

salary_data['gender']=encoder.fit_transform(salary_data['gender'])

salary_data.head()

salary_data['workclass']=encoder.fit_transform(salary_data['workclass'])

salary_data.head()

salary_data['marital-status']=encoder.fit_transform(salary_data['marital-status'])
salary_data['occupation']=encoder.fit_transform(salary_data['occupation'])
salary_data['relationship']=encoder.fit_transform(salary_data['relationship'])
salary_data['race']=encoder.fit_transform(salary_data['race'])
salary_data['native-country']=encoder.fit_transform(salary_data['native-country'])


salary_data.head()

salary_data.isnull().sum()

x=salary_data.drop(columns='income',axis=1)

y=salary_data['income']

print(x)

print(y)

y=y.replace({'<=50K':0,'>50K':1})

 <=50K = 0

 >50K = 1

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x=scaler.fit_transform(x)

print(x)

print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23,stratify=y)

from sklearn.metrics import accuracy_score

print(x.shape,x_train.shape,x_test.shape)

from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(solver='adam',hidden_layer_sizes=(5,2),random_state=2,max_iter=2000)
clf.fit(x_train,y_train)
predict2=clf.predict(x_test)
prediction=accuracy_score(predict2,y_test)
print("acuracy score of the test data: ",prediction)

import joblib

joblib.dump(clf, 'model.pkl')


