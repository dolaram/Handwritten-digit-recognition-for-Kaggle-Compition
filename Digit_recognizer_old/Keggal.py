#Kegal Titanic
#import Modules
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
#Read Csv
df=pd.read_csv('train.csv')
df2=pd.read_csv('test.csv')
output=df.Survived
np.random.seed(0)
df['Norm_Sex']=pd.factorize(df['Sex'])[0]
df['Norm_Embarked']=pd.factorize(df['Embarked'])[0]

df['Norm_Age']=df.Age.fillna(value=0)
x,y=df.Name.str.split(',').str.get(0), df.Name.str.split(',').str.get(1)
y,z=y.str.split('.').str.get(0), y.str.split('.').str.get(1)
df['Norm_Name']=pd.factorize(y)[0]
x1,y1=df.Ticket.str.split(' ').str.get(0), df.Ticket.str.split(' ').str.get(1)
df=df.drop(['PassengerId', 'Survived','Name','Sex','Age','Ticket','Cabin','Embarked'], axis=1)
features=df.columns
clf= RandomForestClassifier(n_estimators=1000, criterion='gini', random_state=0, n_jobs=-1)
X_train, X_test, y_train, y_test= train_test_split(df, output, random_state=0)
clf.fit(X_train, y_train)
n=range(1,1000)
t1=[]
t2=[]
t3=[]

# =============================================================================
# for i in range (1,100):
#     clf.predict(X_test)
#     clf= RandomForestClassifier(n_estimators=i, criterion='gini', random_state=0, max_features='auto', n_jobs=-1)
#     clf.fit(X_train, y_train)
#     t1.append(clf.score(X_test, y_test))
#     t2.append(clf.score(X_train, y_train))
#     print()
#     t3.append(clf.score(X_train, y_train)-clf.score(X_test, y_test))
# plt.plot(n, t1)
# =============================================================================


print(clf.score(X_test, y_test))

df2['Norm_Sex']=pd.factorize(df2['Sex'])[0]
df2['Norm_Embarked']=pd.factorize(df2['Embarked'])[0]
df2['Norm_Age']=df2.Age.fillna(value=0)
x,y=df2.Name.str.split(',').str.get(0), df2.Name.str.split(',').str.get(1)
y,z=y.str.split('.').str.get(0), y.str.split('.').str.get(1)
df2['Norm_Name']=pd.factorize(y)[0]
x1,y1=df2.Ticket.str.split(' ').str.get(0), df2.Ticket.str.split(' ').str.get(1)
df2=df2.drop(['PassengerId','Name','Sex','Age','Ticket','Cabin','Embarked'], axis=1)
features1=df2.columns
df2=df2.fillna(0)
op=clf.predict(df2[features])
im=list(zip(df[features], clf.feature_importances_))
df2['Survived']=op
df2.to_csv('final.csv')

    