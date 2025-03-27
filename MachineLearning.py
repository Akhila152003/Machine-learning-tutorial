#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report


# # Load dataset

# In[16]:


df = pd.read_csv("AmazonSalesData.csv")


# # Display basic info

# In[17]:


print(df.info())
print(df.describe())


# # Check for missing values

# In[18]:


print(df.isnull().sum())


# In[19]:


# Display first few rows
print(df.head())


# # Convert categorical data to numerical using Label Encoding

# In[20]:


le = LabelEncoder()
df['Region'] = le.fit_transform(df['Region'])
df['Country'] = le.fit_transform(df['Country'])
df['Item Type'] = le.fit_transform(df['Item Type'])
df['Sales Channel'] = le.fit_transform(df['Sales Channel'])  # Online = 1, Offline = 0
df['Order Priority'] = le.fit_transform(df['Order Priority'])


# # Convert date columns to datetime

# In[21]:


df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])


# # Create a new feature: Delivery Time (Days between Order and Ship)

# In[22]:


df['Delivery Time'] = (df['Ship Date'] - df['Order Date']).dt.days


# # Drop unnecessary columns

# In[23]:


df = df.drop(['Order ID', 'Order Date', 'Ship Date'], axis=1)


# # Normalize numerical features

# In[24]:


scaler = StandardScaler()
num_cols = ['Units Sold', 'Unit Price', 'Unit Cost', 'Total Revenue', 'Total Cost', 'Total Profit', 'Delivery Time']
df[num_cols] = scaler.fit_transform(df[num_cols])


# In[25]:


print(df.head())


# # Distribution of Total Profit

# In[26]:


plt.figure(figsize=(8,5))
sns.histplot(df['Total Profit'], bins=50, kde=True, color='blue')
plt.title('Distribution of Total Profit')
plt.show()


# # Sales Channel Distribution (Online vs Offline)

# In[27]:


plt.figure(figsize=(6,4))
sns.countplot(x=df['Sales Channel'])
plt.title('Sales Channel Distribution (Online vs Offline)')
plt.show()


# # Profit Distribution per Item Type

# In[28]:


plt.figure(figsize=(10,5))
sns.boxplot(x='Item Type', y='Total Profit', data=df)
plt.xticks(rotation=90)
plt.title('Profit Distribution per Item Type')
plt.show()


# # Correlation Heatmap

# In[29]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# # Total Revenue vs Total Cost

# In[30]:


plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Total Revenue'], y=df['Total Cost'], alpha=0.5)
plt.title('Total Revenue vs Total Cost')
plt.show()


# # Order Priority Count

# In[31]:


plt.figure(figsize=(6,4))
sns.countplot(x='Order Priority', data=df)
plt.title('Order Priority Count')
plt.show()


# # Random Forest Regressor for Total Profit Prediction

# In[32]:


# Define features and target variable
X_reg = df.drop(columns=['Total Profit', 'Sales Channel'])
y_reg = df['Total Profit']


# In[33]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Predictions
y_pred = regressor.predict(X_test)


# # Evaluate model

# In[34]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)


# In[35]:


print("Random Forest Regressor Performance:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


# # Model 2: Random Forest Classifier for Sales Channel Prediction

# In[36]:


# Define features and target variable
X_cls = df.drop(columns=['Sales Channel'])
y_cls = df['Sales Channel']


# In[37]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Train model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)


# In[38]:


# Predictions
y_pred = classifier.predict(X_test)


# # Evaluate model

# In[39]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Classifier Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[ ]:




