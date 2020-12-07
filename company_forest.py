import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split



company_forest=pd.read_csv("C:/Users/W R VARUN/Desktop/Data Science/Random_Forest/Company_Random_Forest.csv")

"----------------------------------------EDA-----------------------------------------------------------------------"

#GETTING ALL THE RELEVENT DATA ABOUT DATSET
profile=ProfileReport(company_forest,explorative=True)
#print(profile)

#PLOT FOR SEEING NULL VALUES
fig=plt.subplots(figsize=(10,10))
count=company_forest.isnull().sum()
#print(count)
sns.lineplot(company_forest.columns,count)
plt.show()


#UNUSUAL VALUES DEALING WITH MEAN VALUES
company_forest.loc[company_forest['Advertising'].values==0,'Advertising']=company_forest['Advertising'].mean()
company_forest.loc[company_forest['Sales'].values==0,'Sales']=company_forest['Sales'].mean()

#DIVIDING THE TARGET VALUES AS LOW & HIGH SALES
company_forest["Sales"]=np.where(company_forest["Sales"]<8,'Low_Sales','High_Sales')

#CREATING DUMMIES FOR CATEGORICAL VALUES
dummy=pd.get_dummies(company_forest[['Urban','US']])

#CONCATINATING DUMMIES & ORIGINAL DATASET
company_forest_final=pd.concat([company_forest,dummy],axis=1)
company_forest_final=company_forest_final.drop(['Urban','US'],axis=1)
#print(company_forest_final)

#CONVERTING ONE OF THE COLUMN SO AS TO DO LABEL ENCODING
company_forest_final['ShelveLoc']=company_forest_final['ShelveLoc'].replace(to_replace=['Bad','Medium','Good'],value=['A','B','C'])
#print(company_forest_final)

#LABEL ENCODING 
lab=LabelEncoder()
company_forest_final['ShelveLoc']=lab.fit_transform(company_forest_final['ShelveLoc'])
company_forest_final['ShelveLoc']
#print(company_forest_final)


"--------------------------------------------MODELLING-----------------------------------------------------------"

# 
X=company_forest_final.iloc[:,1:]
Y=company_forest_final.iloc[:,0]

#SPLITTING THE DATASET IN TRAIN & TEST
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=50,test_size=0.25,stratify=Y)

#MODEL1
#CREATING BASEMODEL WITH DEFAULT PARAMETERS
forest=RandomForestClassifier().fit(x_train,y_train)
#print(forest.score(x_train,y_train),'Train_Score')    #100%
#print(forest.score(x_test,y_test),'Test_score')	   #77%
#print('\n')
pred1=forest.predict(x_test)
report_model1=classification_report(pred1,y_test)		#MODEL IS OVERFITTING
#print(report_model1)


#MODEL2

#USING VALIDATION TECHNIQUE
strf=StratifiedKFold(n_splits=10,shuffle=True,random_state=50)

acc_train=[]
acc_test=[]

#HYPERPARAMTER OPTIMIZATION DONE FOR BETTER PERFORMANCE
forest1=RandomForestClassifier(bootstrap=True,oob_score=True,n_jobs=-1,n_estimators=89,max_depth=15,max_features=6,min_samples_leaf=5,min_samples_split=8,max_samples=0.6)

#PLOTTING FOR FEATURES IMPORTANCE
figure=plt.subplots(figsize=(10,10))
sns.barplot(x=X.columns,y=forest1.feature_importances_)
plt.xlabel('Features')
plt.ylabel('Relative_Importance')
plt.title('Features Importance in Random Forest')


for x_train_new,x_test_new in strf.split(X,Y):

    x_train,x_test=X.iloc[x_train_new],X.iloc[x_test_new]
    y_train,y_test=Y.iloc[x_train_new],Y.iloc[x_test_new]

    forest1.fit(x_train,y_train)
    pred_model2=forest1.predict(x_test)
    #print(forest1.score(x_train,y_train))
    #print(forest1.score(x_test,y_test))
    
    acc_train.append(forest1.score(x_train,y_train))
    acc_test.append(forest1.score(x_test,y_test))
 
    report_model2=classification_report(y_test,pred_model2)
    #print(report_model2)

#MEAN ACCUARCY OF TRAIN   
#print("MEAN accuracy of final model: ",np.array(acc_train).mean())			#91% ACCURACY

#MEAN ACCUARCY OF TEST   
#print("MEAN accuracy of final model: ",np.array(acc_test).mean())			#82.5% ACCURACY

"----------------------------------------------------------------------------------------------------------------"
