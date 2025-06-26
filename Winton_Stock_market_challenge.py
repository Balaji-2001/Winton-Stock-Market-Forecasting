#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Load dataset (update with your path or use Colab Drive/Kaggle)
train_df = pd.read_csv("D:/Projects/Winton Stock Market Dataset/train.csv")
test_df = pd.read_csv("D:/Projects/Winton Stock Market Dataset/test.csv")

# Feature and target columns
features = [f'Feature_{i}' for i in range(1, 26)] + ['Ret_MinusTwo', 'Ret_MinusOne']
targets = ['Ret_PlusOne', 'Ret_PlusTwo']

# Input features and targets
X_train = train_df[features]
y_train = train_df[targets]
X_test = test_df[features]

# Preprocessing and Model pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', MultiOutputRegressor(LinearSVR(random_state=42)))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = pipeline.predict(X_test)

# (Optional) Predict on train and evaluate WMAE
y_pred_train = pipeline.predict(X_train)
train_weights = train_df['Weight_Daily']
mae_plus1 = np.average(np.abs(y_pred_train[:, 0] - y_train['Ret_PlusOne']), weights=train_weights)
mae_plus2 = np.average(np.abs(y_pred_train[:, 1] - y_train['Ret_PlusTwo']), weights=train_weights)
wmae = (mae_plus1 + mae_plus2) / 2
print(f'WMAE: {wmae:.6f}')

# Prepare submission
submission = []
for i in range(len(y_pred)):
    row = []
    for j in range(1, 61):
        row.append((f"{i+1}_{j}", 0))  # Dummy predictions for D-steps
    row.append((f"{i+1}_61", y_pred[i][0]))  # D+1 prediction
    row.append((f"{i+1}_62", y_pred[i][1]))  # D+2 prediction
    submission.extend(row)

submission_df = pd.DataFrame(submission, columns=["Id", "Predicted"])
submission_df.to_csv("new_submission.csv", index=False)
print("Submission saved.")


# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error


# In[4]:


# Load dataset (update with your path or use Colab Drive/Kaggle)
train= pd.read_csv("D:/Projects/Winton Stock Market Dataset/train.csv")
test = pd.read_csv("D:/Projects/Winton Stock Market Dataset/test.csv")


# In[5]:


# 2. Preprocessing & Feature Engineering
def preprocess(df):
    # Fill missing values
    df = df.fillna(df.median())
    # Example: create intraday mean and std features from minute returns
    ret_cols = [col for col in df.columns if 'Ret_' in col]
    df['intraday_mean'] = df[ret_cols].mean(axis=1)
    df['intraday_std'] = df[ret_cols].std(axis=1)
    # Example: interaction feature
    if 'Feature_1' in df.columns:
        df['f1_x_std'] = df['Feature_1'] * df['intraday_std']
    return df

train = preprocess(train)
test = preprocess(test)


# In[6]:


# 3. Select Features and Targets
feature_cols = [col for col in train.columns if col.startswith('Feature_') or 'intraday' in col or 'f1_x_std' in col]
X = train[feature_cols]
# Predicting returns for D+1 and D+2 as example targets
y = train[['Ret_PlusOne', 'Ret_PlusTwo']]


# In[7]:


# 4. Scale Features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test[feature_cols])


# In[8]:


# 5. Train/Test Split for Validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[9]:


# 6. Build and Train MultiOutputRegressor Model
base_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)


# In[11]:


# 7. Validation (optional)
y_pred = model.predict(X_val)
print("Validation MAE:", mean_absolute_error(y_val, y_pred))


# In[12]:


# 8. Predict on Test Data
test_preds = model.predict(X_test_scaled)


# In[13]:


# 9. Prepare Submission
submission = pd.DataFrame(test_preds, columns=['Ret_PlusOne', 'Ret_PlusTwo'])
# If you need to match a sample submission format, adjust accordingly:
# submission['Id'] = test['Id']
# submission.to_csv('submission.csv', index=False)


# In[14]:


print(submission.head())


# In[15]:


submission.head(20)


# In[18]:


import os


# In[20]:


# simply:
submission.to_csv("winton stock market submission.csv", index=False)
print("Saved to", os.path.abspath("winton_stock_market_submission.csv"))


# In[ ]:




