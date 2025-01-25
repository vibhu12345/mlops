# %%
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# %%
# Load the dataset
url = "https://raw.githubusercontent.com/dataprofessor/data/master/heart-disease-cleveland.csv"
data = pd.read_csv(url)

# %%
# Inspect the dataset
# Convert all columns to numeric, setting invalid entries to NaN
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna()
print("Dataset Overview:")
print(data.head())


# %%
print("\nSummary:")
print(data.info())

# %%
# Data Preprocessing
# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())
#Trim columnn nanmes
data.columns = data.columns.to_series().apply(lambda x: x.strip())
#Simplify Diagnosis column
data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x > 0 else 0)

# %%
# Feature Scaling
scaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

# %%
# Split the dataset
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)


# %%
# Evaluate the model
y_pred = model.predict(X_test)

# %%
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# %%
#Save the model
with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as 'heart_disease_model.pkl'")


