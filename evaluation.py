import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel

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
    duplicates = cols[cols.duplicated()].unique()
    if len(duplicates) > 0:
        for dup in duplicates:
            indices = cols[cols == dup].index.values.tolist()
            for i, index in enumerate(indices):
                if i != 0:
                    cols[index] = dup + '_' + str(i)
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
#categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
#numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()

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
feature_names = preprocessor.get_feature_names_out() # Get the names after column transformation

# 10. Transform the testing data
X_test_processed = preprocessor.transform(X_test)

# 11. Randomized Search and Set hyperparameters
param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],  # Number of boosting rounds
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],  # Step size shrinkage
    'max_depth': [3, 4, 5, 6, 7, 8],  # Maximum depth of a tree
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  # Subsample ratio of the training instance
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],  # Subsample ratio of columns when constructing each tree
    'gamma': [0, 0.1, 0.2, 0.3, 0.4], # Minimum loss reduction required to make a further partition on a leaf node
    'reg_alpha': [0.001, 0.01, 0.1, 1], # L1 regularization term on weights
    'reg_lambda': [0.001, 0.01, 0.1, 1] # L2 regularization term on weights
}

# Set up RandomizedSearchCV
# Scale_pos_weight has been predecided
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# Perform cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#Base Model has been defined in xgb_model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)

randomized_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=10, # Number of parameter combinations to try
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1 #Higher number results in function details being printed out
)

# Fit RandomizedSearchCV to the training data
randomized_search.fit(X_train_processed, y_train)

# Print best parameters and score
print("Best parameters found: ", randomized_search.best_params_)
print("Best ROC AUC score: ", randomized_search.best_score_)

# Get the best model from RandomizedSearchCV
best_xgb_model = randomized_search.best_estimator_

# 14. Apply Feature Selection
# Feature Selection

# Create a SelectFromModel object
selection_threshold = 0.01 #Set an importance threshold.
select = SelectFromModel(best_xgb_model, threshold=selection_threshold, prefit=True)

# Transform the training and testing data
select.fit(X_train_processed, y_train)
X_train_selected = select.transform(X_train_processed)
X_test_selected = select.transform(X_test_processed)

# Get the selected features
selected_feat = feature_names[select.get_support()]

print('total features: {}'.format((X_train_processed.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with importance greater than {}:'.format(selection_threshold))
print(selected_feat)

#Reassign best_xgb_model because fit must be called and then used

best_xgb_model.fit(X_train_selected, y_train)

# 15. Evaluate the model on the test set
y_pred = best_xgb_model.predict(X_test_selected)
y_prob = best_xgb_model.predict_proba(X_test_selected)[:, 1]

# Print evaluation metrics
print("\n--- Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot feature importances
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(best_xgb_model.feature_importances_, index = selected_feat)
feat_importances.nlargest(20).plot(kind='barh') #Displaying top 20
plt.title("Top 20 Feature Importances")
plt.show()

# Generate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot ROC curve
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_prob))
plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()