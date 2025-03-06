import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
file_path = r"C:\Users\ludwi\OneDrive\Desktop\insider-threat-project\f.xlsx"
email_path = r"C:\Users\ludwi\OneDrive\Desktop\insider-threat-project\e.xlsx"
device_path = r"C:\Users\ludwi\OneDrive\Desktop\insider-threat-project\d.xlsx"
logon_path = r"C:\Users\ludwi\OneDrive\Desktop\insider-threat-project\l.xlsx"

# 1. Load the data
try:
    file_df = pd.read_excel(file_path)
    email_df = pd.read_excel(email_path)
    device_df = pd.read_excel(device_path)
    logon_df = pd.read_excel(logon_path)
except FileNotFoundError as e:
    print(f"Error: One or more files not found. Please check the file paths.\n{e}")
    exit()

# ---  Date/Time Feature Engineering and Preparation ---

def prepare_data_with_time_features(df, time_window='D'):
    """
    Prepares the data by:
        1.  Ensuring 'date' is datetime.
        2.  Creating a 'time_window' column.
        3.  Extracting various date/time features.
        4.  Returns the dataframe with extracted features.
    """
    try:
        #Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        #Create time_window
        df['time_window'] = df['date'].dt.to_period(time_window)
        df['time_window'] = df['time_window'].astype(str)

        # Extract date/time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
        df['dayofyear'] = df['date'].dt.dayofyear
        df['quarter'] = df['date'].dt.quarter
        df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)

        return df
    except Exception as e:
        print(f"Error in prepare_data_with_time_features: {e}")
        return None # Or handle the error as appropriate

# Apply the time feature extraction
file_df = prepare_data_with_time_features(file_df)
email_df = prepare_data_with_time_features(email_df)
device_df = prepare_data_with_time_features(device_df)
logon_df = prepare_data_with_time_features(logon_df)

#Aggregate Features
# Define columns to group by (user, time_window)
groupby_cols = ['user', 'time_window']
# Identify columns to aggregate (excluding identifiers and text data)
file_numeric_cols = file_df.select_dtypes(include=np.number).columns.tolist()
file_categorical_cols = file_df.select_dtypes(include='object').columns.tolist()

device_numeric_cols = device_df.select_dtypes(include=np.number).columns.tolist()
device_categorical_cols = device_df.select_dtypes(include='object').columns.tolist()

email_numeric_cols = email_df.select_dtypes(include=np.number).columns.tolist()
email_categorical_cols = email_df.select_dtypes(include='object').columns.tolist()

logon_numeric_cols = logon_df.select_dtypes(include=np.number).columns.tolist()
logon_categorical_cols = logon_df.select_dtypes(include='object').columns.tolist()

# Ensure date column is not passed to aggregate function
for cols in [file_numeric_cols, device_numeric_cols, email_numeric_cols, logon_numeric_cols]:
    if 'date' in cols:
        cols.remove('date')

def aggregate_features(df, groupby_cols, numeric_cols, categorical_cols):
    """Aggregates features within each time window, handling both numeric and categorical."""
    if df is None or df.empty:
        print("Warning: DataFrame is empty or None. Returning empty DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame

    #Numeric aggregation
    numeric_agg = df.groupby(groupby_cols)[numeric_cols].agg(['mean', 'std', 'sum'])
    numeric_agg.columns = ['_'.join(col).strip() for col in numeric_agg.columns]
    numeric_agg = numeric_agg.fillna(0) # Fill NaNs after aggregation

    #Categorical aggregation - count the occurences
    def safe_mode(x):
        counts = x.value_counts()
        if not counts.empty:
            return counts.index[0]
        else:
            return 'missing'  # Or another suitable default value

    categorical_agg = df.groupby(groupby_cols)[categorical_cols].agg(safe_mode)
    categorical_agg = categorical_agg.fillna('missing')

    aggregated_df = pd.concat([numeric_agg, categorical_agg], axis = 1).reset_index()
    return aggregated_df

#Aggregate Features
file_features = aggregate_features(file_df, groupby_cols, file_numeric_cols, file_categorical_cols)
device_features = aggregate_features(device_df, groupby_cols, device_numeric_cols, device_categorical_cols)
email_features = aggregate_features(email_df, groupby_cols, email_numeric_cols, email_categorical_cols)
logon_features = aggregate_features(logon_df, groupby_cols, logon_numeric_cols, logon_categorical_cols)

# Add print statements to check DataFrame columns BEFORE merging
print("Columns in file_features:", file_features.columns)
print("Columns in device_features:", device_features.columns)
print("Columns in email_features:", email_features.columns)
print("Columns in logon_features:", logon_features.columns)

# Add a check for empty DataFrames before merging
if file_features.empty or device_features.empty or email_features.empty or logon_features.empty:
    print("Warning: One or more aggregated DataFrames are empty.  Check the aggregation and filtering steps.")
    exit()

# 2. Merge the aggregated features into a single DataFrame
try:
    all_features = file_features.merge(device_features, on=['user', 'time_window'], how='outer', suffixes=('_file', '_device'))
    all_features = all_features.merge(email_features, on=['user', 'time_window'], how='outer', suffixes=('_email', '_logon'))
    all_features = all_features.merge(logon_features, on=['user', 'time_window'], how='outer', suffixes=('_logon', '_log'))
except KeyError as e:
    print(f"KeyError during merge: {e}")
    print("Check if 'user' and 'time_window' are present in ALL aggregated DataFrames.")
    exit()

# Set 'user' as index for better interpretability
all_features = all_features.set_index('user')

# 3. Define features to preprocess
numerical_features = all_features.select_dtypes(include=np.number).columns.tolist()
categorical_features = all_features.select_dtypes(include='object').columns.tolist()

#Remove user and time_window from numerical_features or categorical_features
to_remove = ['user', 'time_window']

for feature in to_remove:
    if feature in numerical_features:
        numerical_features.remove(feature)
    if feature in categorical_features:
        categorical_features.remove(feature)

# The all_features dataframe now has the preprocessed data. You can save it or proceed to train a model.
print(all_features.head())
# Separate features and the target
X = all_features.copy()  # Use all features in all_features. Use copy() to prevent modification
print(X.head())

# Reset Index and create y before preprocessing
X = X.reset_index()
print(X.head())

#Create an empty target column
y = pd.Series([0] * len(X))
print(X.head())
#Mark some users as insider threats
insider_threats = ['ABC0174','AOK0844','ATE0869']

# check that index exist in X
y[X['user'].isin(insider_threats)] = 1

# 7. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

# 8. Store and then drop the user column
user_train = X_train['user']
user_test = X_test['user']
X_train = X_train.drop('user', axis = 1)
X_test = X_test.drop('user', axis = 1)

def rename_duplicate_columns(df):
    """Renames duplicate columns in a DataFrame by adding a counter suffix."""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

# Rename duplicate columns in X_train and X_test
X_train = rename_duplicate_columns(X_train)
X_test = rename_duplicate_columns(X_test)

# 4. Create preprocessing pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with the median
    ('scaler', MinMaxScaler())  # Scale numerical features to the range [0, 1]
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute with most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# 5. Combine pipelines using ColumnTransformer
# Get the final list of categorical columns AFTER column renaming
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()

# Create the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_cols)
    ],
    remainder='passthrough'
)
# 9. Fit and transform the training data
X_train_processed = preprocessor.fit_transform(X_train)

# 10. Transform the testing data
X_test_processed = preprocessor.transform(X_test)

# 11. Initialize and train XGBoost model
# Handle imbalanced data with scale_pos_weight
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)  # Ratio of negative to positive samples
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)  # Add scale_pos_weight

# 12. Cross-validation (optional, but recommended)
# Stratified K-Fold cross-validation to handle imbalanced data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(xgb_model, X_train_processed, y_train, cv=cv, scoring='roc_auc')
print("Cross-validation ROC AUC scores:", scores)
print("Mean cross-validation ROC AUC score:", scores.mean())

# 13. Train the model
xgb_model.fit(X_train_processed, y_train)

# 14. Make predictions on the test set
y_pred = xgb_model.predict(X_test_processed)
y_prob = xgb_model.predict_proba(X_test_processed)[:, 1]  # Probabilities for the positive class

# --- Evaluation ---
print("\n--- Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# --- Visualization (Optional) ---
# Feature Importance
feature_importances = xgb_model.feature_importances_
#feature_names = X_train.columns  # Or use preprocessor.get_feature_names_out()
feature_names = preprocessor.get_feature_names_out()

# Create a dataframe for feature importances
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Print feature importances
print("\nFeature Importances:")
print(feature_importance_df.head(10))

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Feature Importances')
plt.show()