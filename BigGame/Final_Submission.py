import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
np.set_printoptions(suppress=True)


infile=r'C:\PraveenT\PycharmProjects\Trial\Hackerearth\BigGame\train.csv'
testfile=r'C:\PraveenT\PycharmProjects\Trial\Hackerearth\BigGame\test.csv'
sample=r'C:\PraveenT\PycharmProjects\Trial\Hackerearth\BigGame\sample_submission.csv'
outfile=r'C:\PraveenT\PycharmProjects\Trial\Hackerearth\BigGame\Output_Upload\output.csv'
compfile=r'C:\PraveenT\PycharmProjects\Trial\Hackerearth\BigGame\comp.csv'
trialfile=r'C:\PraveenT\PycharmProjects\Trial\Hackerearth\BigGame\trial.csv'
teststack=r'C:\PraveenT\PycharmProjects\Trial\Hackerearth\BigGame\teststack.csv'

df=pd.read_csv(infile)
fun={'five':5,'four':4,'six':6,'three':3,'seven':7,'eight':8,'two':2,'nine':9,'one':1,'ten':10}
fun1={'Intermediate':2,'Beginner':3,'Advanced':1}
fun2={'Less_Than_Four_Billion':3,'Above_Four_Billion':4, 'Less_Than_Three_Billion':3}
fun3={'Balanced':2,'Aggressive_Offense':3,'Aggressive_Defense':1,'Relaxed':4}
df.Number_Of_Injured_Players=df.Number_Of_Injured_Players.map(fun)
df['Coach_Experience_Level']=df['Coach_Experience_Level'].map(fun1)
#df.Team_Value=df.Team_Value.map(fun2)
#df.Playing_Style=df.Playing_Style.map(fun3)

#X=df.drop(['ID','Won_Championship'],axis=1).values
X=df.drop(['Team_Value','Playing_Style','ID','Won_Championship'],axis=1).values
y=df['Won_Championship'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10,stratify=y,test_size=0.2)

mod=pd.read_csv(compfile)
mod['y_test']=y_test

model1=XGBClassifier(learning_rate=0.3,n_estimators=200,reg_lambda=0,min_child_weight=3)
model1.fit(X_train,y_train)
m1_pred=model1.predict(X_test)
mod['xgb']=m1_pred
f=f1_score(y_test, m1_pred)
print('XGB: '+str(f1_score(y_test,m1_pred)))

model2=DecisionTreeClassifier(max_depth=9,min_samples_leaf=5)
model2.fit(X_train,y_train)
m2_pred=model2.predict(X_test)
mod['dt']=m2_pred
print('DT: '+str(f1_score(y_test,m2_pred)))

model3=SVC(kernel='rbf',gamma=0.4)
model3.fit(X_train,y_train)
m3_pred=model3.predict(X_test)
mod['svc']=m3_pred
f=f1_score(y_test,m3_pred)
print('SVM : '+str(f1_score(y_test,m3_pred)))

model4=RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,min_samples_split=11)
model4.fit(X_train,y_train)
m4_pred=model4.predict(X_test)
mod['rfc']=m4_pred
f=f1_score(y_test,m4_pred)
print('RFC : '+str(f1_score(y_test,m4_pred)))

model5=GradientBoostingClassifier(learning_rate=0.01,max_features=3,n_estimators=1200,max_depth=5,min_samples_leaf=20,min_samples_split=2)
model5.fit(X_train,y_train)
m5_pred=model4.predict(X_test)
mod['gb']=m5_pred
f=f1_score(y_test,m5_pred)
print('GB : '+str(f1_score(y_test,m5_pred)))
mod['gb']=m5_pred


v=VotingClassifier(estimators=[('xgb',model1),('dt',model2),('svc',model3),('rfc',model4),('gb',model5)],voting='hard')
v.fit(X_train,y_train)
pred=v.predict(X_test)
mod['voting']=pred
mod.to_csv(trialfile,index=False)
print('Accuracy before Stack: '+str(accuracy_score(y_test,pred)))
print('F1 Score before Stack : '+str(f1_score(y_test,pred)))


stk=pd.read_csv(trialfile)

X_stk=stk.iloc[:,1:5]
y_stk=stk.iloc[:,0]


lr=GradientBoostingClassifier(max_depth=3,min_samples_split=2,min_samples_leaf=5)
lr.fit(X_stk,y_stk)
pred_stk=np.round(lr.predict(X_stk),0)

print('Accuracy after Stack : '+str(accuracy_score(y_test,pred_stk)))
print('F1 Score after Stack : '+str(f1_score(y_test,pred_stk)))


test=pd.read_csv(testfile)
fun={'five':5,'four':4,'six':6,'three':3,'seven':7,'eight':8,'two':2,'nine':9,'one':1,'ten':10}
fun1={'Intermediate':2,'Beginner':3,'Advanced':1}
fun2={'Less_Than_Four_Billion':3,'Above_Four_Billion':4, 'Less_Than_Three_Billion':3}
fun3={'Balanced':2,'Aggressive_Offense':3,'Aggressive_Defense':1,'Relaxed':4}
test.Number_Of_Injured_Players=test.Number_Of_Injured_Players.map(fun)
test['Coach_Experience_Level']=test['Coach_Experience_Level'].map(fun1)


test_X=test.drop(['Team_Value','Playing_Style','ID'],axis=1).values

test_xgb_pred=model1.predict(test_X)
test_dt_pred=model2.predict(test_X)
test_svm_pred=model3.predict(test_X)
test_rfc_pred=model4.predict(test_X)
test_gb_pred=model5.predict(test_X)
test_v_pred=v.predict(test_X)

mod1=pd.read_csv(compfile)
mod1['xgb']=test_xgb_pred
mod1['dt']=test_dt_pred
mod1['svc']=test_svm_pred
mod1['rfc']=test_rfc_pred
mod1['gb']=test_gb_pred
mod1['voting']=test_v_pred

test_X_stk=mod1.iloc[:,1:5]

test_pred_stk=np.round(lr.predict(test_X_stk),0)


submis=pd.read_csv(testfile,usecols=['ID'])
submis['ID']=test['ID']
submis['Won_Championship']=test_pred_stk
submis.to_csv(outfile,index=False)

