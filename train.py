# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import optuna

# Load the dataset downloaded from https://raw.githubusercontent.com/dataprofessor/data/master/heart-disease-cleveland.csv
path = "https://raw.githubusercontent.com/dataprofessor/data/master/heart-disease-cleveland.csv" 
data = pd.read_csv(path)

# Data Preprocessing
# Convert all columns to numeric, setting invalid entries to NaN
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop Missing value rows
data = data.dropna()


#Trim columnn nanmes
data.columns = data.columns.to_series().apply(lambda x: x.strip())
#Simplify Diagnosis column
data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x > 0 else 0)

# Feature Scaling
scaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

# Split the dataset
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



def objective(trial):
    # Define the hyperparameters to tune
    C = trial.suggest_loguniform('C', 1e-5, 1e2)
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])

    # Create the model with suggested hyperparameters
    model = LogisticRegression(C=C, solver=solver, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# best model
best_params = study.best_params
model = LogisticRegression(C=best_params['C'], solver=best_params['solver'], random_state=42)
model.fit(X_train, y_train)

#Save the best model
with open('models/heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as 'heart_disease_model.pkl'")


