# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2.Import Decision tree classifier
3.Fit the data in the model
4.Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: TIMMAPURAM YOGEESWAR
RegisterNumber:  212223230233
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
data.head()

![image](https://github.com/user-attachments/assets/65682d61-8ccd-45f9-88d7-481708f85625)

data.info()

![image](https://github.com/user-attachments/assets/f3bb60e0-a616-4bbf-9ba9-a375b5928b6f)

data.isnull().sum()

![image](https://github.com/user-attachments/assets/23e851f4-fb4e-4dfe-b7ef-392bf73afc97)

data value count

![image](https://github.com/user-attachments/assets/010a6280-8a3b-4aea-98b5-62b70f639415)

data.head() for salary

![image](https://github.com/user-attachments/assets/218ee9d8-1106-4ace-85d4-406353530b0b)

x.head()

![image](https://github.com/user-attachments/assets/1f740ea9-7ce4-4622-964a-9b0be96cf80f)

accuracy value

![image](https://github.com/user-attachments/assets/34addd86-8590-4114-8002-897677f1e1f2)

data prediction

![image](https://github.com/user-attachments/assets/b0b0648e-b639-4c04-b8d0-f2db66d0c636)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
