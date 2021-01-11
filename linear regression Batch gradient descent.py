
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv("Salary_Data.csv")

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import *
en=LabelEncoder()
x[:,-1]=en.fit_transform(x[:,-1])

#one=OneHotEncoder(categorical_features=[-1])
#x=one.fit_transform(x).toarray()
#x=x[:,1:]
#
s=StandardScaler()
x=s.fit_transform(x)



def linear(x_train,y_train):

    theta=np.random.rand(1,x_train.shape[1])
    learning_rate=0.01
    y_train=np.array(y_train).reshape(-1,1)
    counter=0
    while(True):
        counter+=1
        old_theta=theta
        theta=theta-learning_rate*(((np.dot(x_train,theta.T))-y_train)*x_train).mean(axis=0)
        check=theta==old_theta
        if(check.all()):
            break
        
    return theta


x=np.append(np.ones((x.shape[0],1)),x,axis=1)

from sklearn.model_selection import *

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=0)

theta=linear(x_train,y_train)

y_pred=x_test.dot(theta.T)

plt.scatter(x_test[:,1],y_test,color='red')
plt.plot(x_test[:,1],y_pred,color='blue')
plt.xlabel('input-values')
plt.ylabel('predicted values')
plt.title('linear regression fitting for 2d')
plt.show()



