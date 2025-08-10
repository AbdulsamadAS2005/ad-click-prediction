import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def removeOutlier(df,col):
    for i in col:
        q1=df[i].quantile(0.25)
        q3=df[i].quantile(0.75)
        IQR=q3-q1
        max=q3+(1.5*IQR)
        min=q1-(1.5*IQR)

        df=df[(df[i]>=min)&(df[i]<=max)]
    return df


data=pd.read_csv("ad_click_dataset.csv")

data.drop(columns=["id","full_name","time_of_day","ad_position"],inplace=True)

for i in data.select_dtypes("object"):
    data[i]=data[i].fillna("Unknown")

for i in data.select_dtypes("float"):
    mode=data[i].mode()
    data[i]=data[i].fillna(mode[0])

to_label=["gender","device_type","browsing_history"]

data.drop_duplicates(inplace=True)

outlier=["age"]

data=removeOutlier(data,outlier)

X=data.iloc[:,:-1]
Y=data['click']

X=pd.get_dummies(data=X,columns=to_label,drop_first=True)
print(data.shape)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=pd.DataFrame(sc.fit_transform(X),columns=X.columns)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=42)

import tensorflow
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout
from keras.callbacks import EarlyStopping



ann=Sequential()
ann.add(Dense(16,input_dim=X.shape[1],activation="relu"))

ann.add(Dense(32,activation="relu"))
ann.add(Dropout(0.3))
ann.add(Dense(16,activation="relu"))
ann.add(Dropout(0.3))
ann.add(Dense(8,activation="relu"))
ann.add(Dense(1,activation="sigmoid"))

ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
ann.fit(xtrain, ytrain, batch_size=20, epochs=50, validation_data=(xtest, ytest),
        callbacks=EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"))
yTestpred=ann.predict(xtest)
yTrainpred=ann.predict(xtrain)

ytrain_pred=[]
ytest_pred=[]

for i in yTrainpred:
    if i>=0.5:
        ytrain_pred.append(1)
    else:
        ytrain_pred.append(0)

for i in yTestpred:
    if i>=0.5:
        ytest_pred.append(1)
    else:
        ytest_pred.append(0)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(ytrain,ytrain_pred)*100,accuracy_score(ytest,ytest_pred)*100)
print(confusion_matrix(ytrain,ytrain_pred))
print(confusion_matrix(ytest,ytest_pred))