
# Import packages
import numpy as np
import pandas as pd
import glob
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
from xml.etree import ElementTree
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

columns = ['DATE', 
           'CUST_ID', 
           'DEVICE', 
           'P/C', 
           '製令編碼', 
           'vendor', 
           'Dut', 
           'PROCESS_ID', 
           '送修時間', 
           'STATUS', 
           'NEW針長', 
           'NOW針長', 
           '已用針長', 
           '送修時已使用TD', 
           '累積TD', 
           '總使用TD', 
           'PM TD', 
           'MTBF達成率', 
           'MTBF判定', 
           '送修原因', 
           '送修備註', 
           '維修成功率']

df = pd.read_excel('data/MTBF-CP/mtbf_more.xlsx', usecols=[ '送修時間',
                                                            'STATUS',
                                                            'P/C',
                                                            'CUST_ID',
                                                            'vendor',
                                                            'Dut',
                                                            'PROCESS_ID',
                                                            'NEW針長', 
                                                            'NOW針長', 
                                                            '已用針長',
                                                            'MTBF判定',
                                                            '維修前針徑(UM)', 
                                                            '維修前共面(UM)', 
                                                            '維修前針長(MILS)', 
                                                            '維修後針徑(UM)', 
                                                            '維修後共面(UM)', 
                                                            '維修後針長(MILS)',
                                                            '送修時已使用TD'])

df['送修時間'] = df['送修時間'].apply(pd.to_datetime)
df = df.sort_values(by='送修時間', ascending=True)
df = df.reset_index(drop=True)
df['上次維修後針徑(UM)'] = np.nan
df['上次維修後共面(UM)'] = np.nan
df['上次維修後針長(MILS)'] = np.nan




# Shift the columns of 維修後 by 1 down and paste to 上次維修後

for pc in df['P/C'].value_counts().index:
        
    df.loc[df['P/C'] == pc, [ '上次維修後針徑(UM)', 
                              '上次維修後共面(UM)', 
                              '上次維修後針長(MILS)']] = \
    df.loc[df['P/C'] == pc, [ '維修後針徑(UM)', 
                              '維修後共面(UM)', 
                              '維修後針長(MILS)']].shift(1).values


# Drop the Null values and remove rows with STATUS=='定期PM'


df = df.dropna()
df = df.loc[(df['STATUS'] != '定期PM') & (df['送修時已使用TD'] > 100000)]


# The difference of 上次維修後針長 & NEW針長

df['上次已使用針長(MILS)'] = df['NEW針長'] - df['上次維修後針長(MILS)']



use_cols = ['CUST_ID',
            'vendor',
            'Dut',
            'PROCESS_ID',
            'NEW針長',
            '上次維修後針徑(UM)',
            '上次維修後共面(UM)',
            '上次維修後針長(MILS)',
            '上次已使用針長(MILS)',
            '送修時已使用TD']


# Map Categorical Features to a numerical value

# Customer ID
cust_id_map = {name:n for name, n in zip(df['CUST_ID'].value_counts().index, list(range(len(df['CUST_ID'].value_counts()))))}
df['CUST_ID'] = df['CUST_ID'].map(cust_id_map)

# Vendor
vendor_map = {name:n for name, n in zip(df['vendor'].value_counts().index, list(range(len(df['vendor'].value_counts()))))}
df['vendor'] = df['vendor'].map(vendor_map)

# Process ID
process_id_map = {name:n for name, n in zip(df['PROCESS_ID'].value_counts().index, list(range(len(df['PROCESS_ID'].value_counts()))))}
df['PROCESS_ID'] = df['PROCESS_ID'].map(process_id_map)

# Dut
df['Dut'] = df['Dut'].replace('C', 5)
df['Dut'] = df['Dut'].replace('G', 7)
df['Dut'] = df['Dut'].astype('int')



# Change all value to numerical
df['NEW針長'] = df['NEW針長'].astype('float')
df['上次維修後針徑(UM)'] = df['上次維修後針徑(UM)'].astype('float')
df['上次維修後共面(UM)'] = df['上次維修後共面(UM)'].astype('float')
df['上次維修後針長(MILS)'] = df['上次維修後針長(MILS)'].astype('float')
df['送修時已使用TD'] = df['送修時已使用TD'].astype('int')



data = df[use_cols]
data.columns = ['CUST_ID',
              'vendor',
              'Dut',
              'PROCESS_ID',
              'NEWL',
              'R',
              'M',
              'FL',
              'NEWL-FL',
              'TD']

data['NEWL'] = data['NEWL'].round(0).astype('int')
data['R'] = data['R'].round(0).astype('int')
data['M'] = data['M'].round(0).astype('int')
data['FL'] = data['FL'].round(0).astype('int')
data['NEWL-FL'] = data['NEWL-FL'].round(0).astype('int')
data['TD'] = np.digitize(data['TD'], bins=np.arange(150000, 600000, 50000))


# Train Test Split


X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.15, random_state=42, shuffle=True)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Define Model

estimator = []
estimator.append(('SVC', SVC(gamma ='auto', probability = True)))
estimator.append(('RF', RandomForestClassifier()))
estimator.append(('XGB', XGBClassifier()))


vot_soft = VotingClassifier(estimators = estimator, voting ='hard')
vot_soft.fit(X_train, y_train)
y_pred = vot_soft.predict(X_test)

print(f'Accuracy: {round(accuracy_score(y_test, y_pred)*100, 1)}%')


cm = confusion_matrix(y_test, y_pred)
sns.set(rc={'figure.figsize':(10.7,7.27)})
sns.heatmap(cm, annot=True, cmap = 'rocket_r', fmt = '.20g')
plt.xlabel("Predicted")
plt.ylabel("Actual")



capture = np.eye(cm.shape[0], cm.shape[1]) + np.eye(cm.shape[0], cm.shape[1], k=-1)
print(f'Hit rate: {round((cm*capture).sum()/cm.sum()*100, 1)}%')