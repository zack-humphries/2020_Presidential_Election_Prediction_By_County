#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import seaborn as sns


# In[2]:


file = "election.xlsx"
df = pd.read_excel(file, dtype={"fips": str})
df.head()


# In[3]:


import plotly.express as px


# In[4]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


# ## 2020 Election Margin Map

# In[5]:


fig = px.choropleth(df, geojson=counties, 
                        locations='fips', 
                        color_continuous_scale="rdbu",
                        color='margin_2020',
                        hover_data=["leader_party_id", "name", "state"],
                        range_color=[-80,80]
                    )
fig.update_geos(fitbounds='locations', visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ## 2016 to 2020 Margin Shift Map

# In[6]:


fig = px.choropleth(df, geojson=counties, 
                        locations='fips', 
                        color_continuous_scale="rdbu",
                        color='shift',
                        hover_data=["name", "state"],
                        range_color=[-25,25]
                    )
fig.update_geos(fitbounds='locations', visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ## Median Personal Income Map

# In[7]:


fig = px.choropleth(df, geojson=counties, 
                        locations='fips', 
                        color='personal_income',
                        hover_data=["name", "state"],
                        range_color=[0,100000]
                    )
fig.update_geos(fitbounds='locations', visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[8]:


df.describe()


# In[9]:


#Preparing data to be analyzed
import numpy as np
df1 = df[["percent_absentee","votes", "personal_income", 
          "education", "land_area", "percent_white", "religious", "margin_2016",
          "margin_2020", "leader_party_binary", "age"]]
df1 = df1.dropna()

df1 = df1[df1['votes'] > 0] 

df1["personal_income"] = np.log(df1["personal_income"])
df1["votes"] = np.log(df1["votes"])
df1["land_area"] = np.log(df1["land_area"])
df1["religious"] = np.log(df1["religious"])
df1["margin_2016"] = (df1["margin_2016"]+100)/2

X = df1[["percent_absentee","votes", "personal_income", "land_area", 
         "education", "percent_white", "religious", "age", 
         "margin_2016"
        ]]
#y = df1["margin_2020"]
y = df1["leader_party_binary"]


# ## Multiple Regression Analysis

# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[11]:


from sklearn.linear_model import LinearRegression
modellin = LinearRegression()

modellin.fit(X_train, y_train)

intercept = modellin.intercept_
coefficient = modellin.coef_


# In[12]:


import statsmodels.api as sm
from scipy import stats
# Code help from JARH from https://stackoverflow.com/a/42677750
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[13]:


#Creating a binary prediction model
predicted_class_reg = modellin.predict(X_test)
predicted_class_reg = np.round(predicted_class_reg, 0)
i = 0
for row in predicted_class_reg:
    if row > 1:
        predicted_class_reg[i] = 1
    elif row < 0:
        predicted_class_reg[i] = 0
    i+=1


# In[14]:


pred = pd.DataFrame(data=predicted_class_reg, columns=["prediction"])


# In[15]:


#Preparing prediction data to be compared to actual data
array1 = []

for row in y_test:
    array1.append(row)
    
actual = pd.DataFrame(data=array1, columns=["leader_party_binary"])


# In[16]:


array2 = y_test.index.values

array3 = list(array2)


# In[17]:


array4 = []
arraycounty = []
arraystate = []

for row in array3:
    value = df._get_value(row, "fips")
    value1 = df._get_value(row, "name")
    value2 = df._get_value(row, "state")
    array4.append(value)
    arraycounty.append(value1)
    arraystate.append(value2)
    
fips1 = pd.DataFrame(data=array4, columns=["fips"])
county1 = pd.DataFrame(data=arraycounty, columns=["county"])
state1 = pd.DataFrame(data=arraystate, columns=["state"])


# In[18]:


array5 = []
i = 0

while i < len(pred):
    if (int(pred._get_value(i, "prediction")) != int(actual._get_value(i, "leader_party_binary"))):
        array5.append(0)
    else:
        array5.append(1)
    i += 1    

correct = pd.DataFrame(data=array5, columns=["correct"])


# In[19]:


result = pd.concat([fips1, state1, county1, pred, actual, correct], axis=1)
result.head()


# In[20]:


#Multiple Regression Correctness Map
fig = px.choropleth(result, geojson=counties, 
                        locations='fips', 
                        color_continuous_scale="YlGn",
                        color='correct',
                        hover_data=["state", "county", "prediction", "leader_party_binary", "prediction"],
                        range_color=[0,1]
                    )
fig.update_geos(fitbounds='locations', visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

from sklearn.metrics import confusion_matrix

cmlin = pd.DataFrame(confusion_matrix(y_test, predicted_class_reg))

cmlin['Total'] = np.sum(cmlin, axis=1)

cmlin = cmlin.append(np.sum(cmlin, axis=0), ignore_index=True)

cmlin.columns = ['Predicted Democrat', 'Predicted Republican', 'Total']

cmlin = cmlin.set_index([['Actual Democrat', 'Actual Republican', 'Total']])

print(cmlin)


# In[21]:


from sklearn.metrics import classification_report

print(classification_report(y_test, predicted_class_reg))

# https://en.wikipedia.org/wiki/Confusion_matrix


# ## Random Forest

# In[22]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model 
rf = RandomForestRegressor(n_estimators= 500)

# Train the model on training data
rf.fit(X_train, y_train);

# Get numerical feature importances
importances = list(rf.feature_importances_)

# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.figure(figsize=(12, 6))
plt.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis
plt.xticks(x_values, list(X_train.columns), rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[23]:


# Use the forest's predict method on the test data and condenses to dummy 1 or 0
predictionrf = rf.predict(X_test)

predictionrf = np.round(predictionrf, 0)
i = 0
for row in predictionrf:
    if row > 1:
        predictionrf[i] = 1
    elif row < 0:
        predictionrf[i] = 0
    i+=1


# In[24]:


#Preparing predicted model data to be compared to actual data
predrf = pd.DataFrame(data=predictionrf, columns=["prediction"])

arrayrf1 = []

for row in y_test:
    arrayrf1.append(row)
    
actualrf = pd.DataFrame(data=arrayrf1, columns=["leader_party_binary"])

arrayrf2 = y_test.index.values

arrayrf3 = list(arrayrf2)

arrayrf4 = []
arraycountyrf = []
arraystaterf = []

for row in arrayrf3:
    value = df._get_value(row, "fips")
    value1 = df._get_value(row, "name")
    value2 = df._get_value(row, "state")
    arrayrf4.append(value)
    arraycountyrf.append(value1)
    arraystaterf.append(value2)
    
fips1rf = pd.DataFrame(data=arrayrf4, columns=["fips"])
county1rf = pd.DataFrame(data=arraycountyrf, columns=["county"])
state1rf = pd.DataFrame(data=arraystaterf, columns=["state"])

arrayrf5 = []
i = 0

while i < len(pred):
    if (int(predrf._get_value(i, "prediction")) != int(actualrf._get_value(i, "leader_party_binary"))):
        arrayrf5.append(0)
    else:
        arrayrf5.append(1)
    i += 1    

correctrf = pd.DataFrame(data=arrayrf5, columns=["correct"])

resultrf = pd.concat([fips1rf, state1rf, county1rf, predrf, actualrf, correctrf], axis=1)
resultrf.head()


# In[25]:


# Random Forest Correctness Map
fig = px.choropleth(resultrf, geojson=counties, 
                        locations='fips', 
                        color_continuous_scale="YlGn",
                        color='correct',
                        hover_data=["state", "county", "prediction", "leader_party_binary", "prediction"],
                        range_color=[0,1]
                    )
fig.update_geos(fitbounds='locations', visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[26]:


# Calculate the absolute errors
errorsrf = abs(predictionrf - y_test)

# Calculate mean absolute percentage error (MAPE)
maperf = 100 * sum(errorsrf)/len(errorsrf)

# Calculate and display accuracy
accuracyrf = 100 - maperf
print('Accuracy:', round(accuracyrf, 2), '%.')


# In[27]:


from sklearn.metrics import confusion_matrix

cmrf = pd.DataFrame(confusion_matrix(y_test, predictionrf))

cmrf['Total'] = np.sum(cmrf, axis=1)

cmrf = cmrf.append(np.sum(cmrf, axis=0), ignore_index=True)

cmrf.columns = ['Predicted Dem', 'Predicted Rep', 'Total']

cmrf = cmrf.set_index([['Actual Dem', 'Actual Rep', 'Total']])

print(cmrf)


# In[28]:


from sklearn.metrics import classification_report

print(classification_report(y_test, predictionrf))

# https://en.wikipedia.org/wiki/Confusion_matrix


# ## Artificial Neural Network

# In[29]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[30]:


# Setting up ANN model
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500)
clf.fit(X_train, y_train.values.ravel())

predictions = clf.predict(X_test)


# In[31]:


# Preparing predicted model data to be compared to actual data
predann = pd.DataFrame(data=predictions, columns=["prediction"])

arrayann1 = []

for row in y_test:
    arrayann1.append(row)
    
actualann = pd.DataFrame(data=arrayann1, columns=["leader_party_binary"])

arrayann2 = y_test.index.values

arrayann3 = list(arrayann2)

arrayann4 = []
arraycountyann = []
arraystateann = []

for row in arrayann3:
    value = df._get_value(row, "fips")
    value1 = df._get_value(row, "name")
    value2 = df._get_value(row, "state")
    arrayann4.append(value)
    arraycountyann.append(value1)
    arraystateann.append(value2)
    
fips1ann = pd.DataFrame(data=arrayann4, columns=["fips"])
county1ann = pd.DataFrame(data=arraycountyann, columns=["county"])
state1ann = pd.DataFrame(data=arraystateann, columns=["state"])

arrayann5 = []
i = 0

while i < len(pred):
    if (int(predann._get_value(i, "prediction")) != int(actualann._get_value(i, "leader_party_binary"))):
        arrayann5.append(0)
    else:
        arrayann5.append(1)
    i += 1    

correctann = pd.DataFrame(data=arrayann5, columns=["correct"])

resultann = pd.concat([fips1ann, state1ann, county1ann, predann, actualann, correctann], axis=1)
resultann.head()


# In[32]:


# Artificial Neural Network Correctness Map
fig = px.choropleth(resultann, geojson=counties, 
                        locations='fips', 
                        color_continuous_scale="YlGn",
                        color='correct',
                        hover_data=["state", "county", "prediction", "leader_party_binary", "prediction"],
                        range_color=[0,1]
                    )
fig.update_geos(fitbounds='locations', visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[33]:


cmrf = pd.DataFrame(confusion_matrix(y_test, predictions))

cmrf['Total'] = np.sum(cmrf, axis=1)

cmrf = cmrf.append(np.sum(cmrf, axis=0), ignore_index=True)

cmrf.columns = ['Predicted Dem', 'Predicted Rep', 'Total']

cmrf = cmrf.set_index([['Actual Dem', 'Actual Rep', 'Total']])

print(cmrf)


# In[34]:


print(classification_report(y_test,predictions))

