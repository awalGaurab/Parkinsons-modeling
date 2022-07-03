import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set() # Setting seaborn as default style even if use only matplotlib

pd.set_option('display.max_columns', None)
data  = pd.read_csv('parkinsons.data')
feat = data.drop(['name','status'],axis = 1)
targ = data['status']

# # Get the shape of the dataset
# print(data.shape)
# print(feat.corr())

# # Get first 5 rows of data
# print(data.head())


# # get idea about data types and presence of null values for each column.
# print(data.info())

# # Check/count total null values for each column.
# print(data.isnull().sum())

# # Get statistical idea of the data
# print(data.describe())

# print(feat.columns)
# print(targ.unique()) # This means our target variable is categorical variable

# /----------------------------------------
# Plot each column graph to get idea about distribution of the data.
# -----------------------------------------/
# sns.pairplot(data=feat.iloc[:,0:10])  # Most of the data are right skewed and have some outliers.
# sns.pairplot(data=feat.iloc[:,10:])
# plt.show()

# /----------------------------------------
# Outlier check
# -----------------------------------------/

# label_name = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)','MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
#        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5','MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
#        'spread2', 'D2', 'PPE']

# fig, axes = plt.subplots(5, 5)

# ax = sns.boxplot(ax=axes[0,0], data = feat.loc[:,label_name[0]])
# ax.set_xticklabels([label_name[0]])

# ax = sns.boxplot(ax=axes[0,1], data = feat.loc[:,label_name[1]])
# ax.set_xticklabels([label_name[1]])

# ax = sns.boxplot(ax=axes[0,2], data = feat.loc[:,label_name[2]])
# ax.set_xticklabels([label_name[2]])

# ax = sns.boxplot(ax=axes[0,3], data = feat.loc[:,label_name[3]])
# ax.set_xticklabels([label_name[3]])

# ax = sns.boxplot(ax=axes[0,4], data = feat.loc[:,label_name[4]])
# ax.set_xticklabels([label_name[4]])

# ax = sns.boxplot(ax=axes[1,0], data = feat.loc[:,label_name[5]])
# ax.set_xticklabels([label_name[5]])

# ax = sns.boxplot(ax=axes[1,1], data = feat.loc[:,label_name[6]])
# ax.set_xticklabels([label_name[6]])

# ax = sns.boxplot(ax=axes[1,2], data = feat.loc[:,label_name[7]])
# ax.set_xticklabels([label_name[7]])

# ax = sns.boxplot(ax=axes[1,3], data = feat.loc[:,label_name[8]])
# ax.set_xticklabels([label_name[8]])

# ax = sns.boxplot(ax=axes[1,4], data = feat.loc[:,label_name[9]])
# ax.set_xticklabels([label_name[9]])

# ax = sns.boxplot(ax=axes[2,0], data = feat.loc[:,label_name[10]])
# ax.set_xticklabels([label_name[10]])

# ax = sns.boxplot(ax=axes[2,1], data = feat.loc[:,label_name[11]])
# ax.set_xticklabels([label_name[11]])

# ax = sns.boxplot(ax=axes[2,2], data = feat.loc[:,label_name[12]])
# ax.set_xticklabels([label_name[12]])

# ax = sns.boxplot(ax=axes[2,3], data = feat.loc[:,label_name[13]])
# ax.set_xticklabels([label_name[13]])

# ax = sns.boxplot(ax=axes[2,4], data = feat.loc[:,label_name[14]])
# ax.set_xticklabels([label_name[14]])

# ax = sns.boxplot(ax=axes[3,0], data = feat.loc[:,label_name[15]])
# ax.set_xticklabels([label_name[15]])

# ax = sns.boxplot(ax=axes[3,1], data = feat.loc[:,label_name[16]])
# ax.set_xticklabels([label_name[16]])

# ax = sns.boxplot(ax=axes[3,2], data = feat.loc[:,label_name[17]])
# ax.set_xticklabels([label_name[17]])

# ax = sns.boxplot(ax=axes[3,3], data = feat.loc[:,label_name[18]])
# ax.set_xticklabels([label_name[18]])

# ax = sns.boxplot(ax=axes[3,4], data = feat.loc[:,label_name[19]])
# ax.set_xticklabels([label_name[19]])

# ax = sns.boxplot(ax=axes[4,0], data = feat.loc[:,label_name[20]])
# ax.set_xticklabels([label_name[20]])

# ax = sns.boxplot(ax=axes[4,1], data = feat.loc[:,label_name[21]])
# ax.set_xticklabels([label_name[21]])

# fig.tight_layout()      
# plt.show()


# /---------------------------------------------------------------------------
# Remove an outliers 
# ----------------------------------------------------------------------------/

label_name = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)','MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
       'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5','MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
       'spread2', 'D2', 'PPE']

for coln in label_name:
    q1 = feat[coln].quantile(0.25)
    q3 = feat[coln].quantile(0.75)
    iqr = q3 - q1

    low_limit = q1 - (1.5 * iqr)
    high_limit = q3 + (1.5 * iqr)
    feat = feat[(feat[coln] > low_limit) | (feat[coln] < high_limit)]
    new_df = pd.DataFrame(feat)

new_df.reset_index(inplace=True)
# print(new_df.head())

# /---------------------------------------------------------------------------
# As datas are in different scale we need to make data of each columns into same scale
# ----------------------------------------------------------------------------/

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(new_df),index=new_df.index,columns=new_df.columns)

# /---------------------------------------------------------------------------
# From the dataset we have to find as we have parkinson disease or not.That means
# this problem is classification specifically binary classification.
# For binary classification we will user logistic regression for now.
# ----------------------------------------------------------------------------/
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np


lg_reg = LogisticRegression(C = 0.1, max_iter = 100, penalty = 'l2', solver = 'lbfgs')

X_train,X_test,y_train,y_test = train_test_split(df_scaled,targ,test_size=0.2,random_state=1)

lg_reg.fit(X_train,y_train)
predict = lg_reg.predict(X_test)

report = accuracy_score(y_test,predict)
clf_report = classification_report(y_test,predict)
cm = confusion_matrix(y_test,predict)

print("score on test: " + str(lg_reg.score(X_test, y_test)))
print("score on train: "+ str(lg_reg.score(X_train, y_train)))

print("Accuracy of our model is %0.2f" %(report)) #92%
print("-----------------------------------------------------")
print(clf_report)

print("-----------------------------------------------------")
print("Confusion matrix : ")
print(cm)

# /---------------------------------------------------------------------------
# Hyperparameter tunning
# ----------------------------------------------------------------------------/
# from sklearn.model_selection import GridSearchCV

# param_grid = [
#     {'penalty': ['l1','l2','elasticnet'],
#     'C': [0.001,0.01,0.1,1,10,25,50,75,100,1000],
#     'solver': [ 'lbfgs', 'liblinear', 'sag', 'saga'],
#     'max_iter': [100,200,500,1000,2500,5000,10000,25000],
# }
# ]

# clf = GridSearchCV(estimator = lg_reg,param_grid = param_grid,cv=10,verbose=True,n_jobs = -1)
# clf.fit(df_scaled,targ)
# print("****************************** Best parameter is ",clf.best_params_)

# From the GridSearch hyper-parameter tuning we got following parameter as best parameter.
# Best parameter is  {'C': 0.1, 'max_iter': 100, 'penalty': 'l2', 'solver': 'lbfgs'}

