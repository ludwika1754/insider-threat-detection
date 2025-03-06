from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

# --- Load Data (Ensure this part is executed before the preprocessing) ---
# Define file paths (replace with your actual paths)
logon_path = "C:/Users/ludwi/OneDrive/Desktop/r4.1/l.xlsx"
email_path = "C:/Users/ludwi/OneDrive/Desktop/r4.1/e.xlsx"
device_path = "C:/Users/ludwi/OneDrive/Desktop/r4.1/d.xlsx"
file_path = "C:/Users/ludwi/OneDrive/Desktop/r4.1/f.xlsx"
psychometric_path = "C:/Users/ludwi/OneDrive/Desktop/r4.1/p.xlsx"

# Load data into pandas DataFrames
try:
    logon_df = pd.read_excel(logon_path)
    email_df = pd.read_excel(email_path)
    device_df = pd.read_excel(device_path)
    file_df = pd.read_excel(file_path)
    psychometric_df = pd.read_excel(psychometric_path)

    print("Data loaded successfully.")

except FileNotFoundError as e:
    print(f"Error: One or more files not found.  Check the paths. {e}")
    exit() # Stop execution if files are missing

except Exception as e:
    print(f"An error occurred while loading data: {e}")
    exit()

# --- Inspect Column Names (CRITICAL) ---
print("\nColumn Names in Logon Data:", logon_df.columns)
print("Column Names in Email Data:", email_df.columns)
print("Column Names in Device Data:", device_df.columns)
print("Column Names in File Data:", file_df.columns)
print("Column Names in Psychometric Data:", psychometric_df.columns)


# --- Feature Engineering (Ensure this part is executed) ---
def create_logon_features(logon_df):
    if logon_df.empty:
        print("Warning: Logon DataFrame is empty. Returning original DataFrame.")
        return logon_df

    # ***CORRECTED COLUMN NAMES***
    date_col = 'date'
    user_col = 'user'
    pc_col = 'pc'
    activity_col = 'activity' #This may be not required but put if in use

    # Ensure time columns are datetime objects
    for col in logon_df.columns:
        if 'date' in col.lower():  # try to identify date/time column
            try:
                logon_df[col] = pd.to_datetime(logon_df[col], errors='coerce') # Convert to datetime, handle errors
            except:
                print(f"Warning: Could not convert column '{col}' to datetime.")

    # 1. Session Duration (Requires separate logon and logoff events)
    # This is more complex because you need to find the logoff event for each logon event
    # and calculate the time difference.  I'm skipping this for now.

    # 2. Time of Day (Logon) - Categorical:  Morning, Afternoon, Evening, Night
    if date_col in logon_df.columns:
        logon_df['logon_hour'] = logon_df[date_col].dt.hour
        def categorize_time(hour):
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 18:
                return 'Afternoon'
            elif 18 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
        logon_df['logon_time_category'] = logon_df['logon_hour'].apply(categorize_time)
        logon_df.drop('logon_hour', axis=1, inplace=True) # Remove the hour column after categorization
    else:
        print("Warning: 'date' column not found in logon data.")

    # 3. Off-Hours Logon (Binary):  1 if logon is outside normal working hours, 0 otherwise
    if date_col in logon_df.columns:
        logon_df['off_hours_logon'] = ((logon_df[date_col].dt.hour < 8) | (logon_df[date_col].dt.hour > 18)).astype(int)
    else:
        print("Warning: 'date' column not found in logon data.")

    return logon_df


def create_email_features(email_df):
    if email_df.empty:
        print("Warning: Email DataFrame is empty. Returning original DataFrame.")
        return email_df

    # ***CORRECTED COLUMN NAMES***
    date_col = 'date'
    user_col = 'user'
    pc_col = 'pc'
    to_col = 'to'
    cc_col = 'cc'
    bcc_col = 'bcc'
    from_col = 'from'
    size_col = 'size'
    attachments_col = 'attachments'
    content_col = 'content'


    # Ensure timestamp column is datetime object
    for col in email_df.columns:
        if 'date' in col.lower():  # try to identify date/time column
            try:
                email_df[col] = pd.to_datetime(email_df[col], errors='coerce') # Convert to datetime, handle errors
            except:
                print(f"Warning: Could not convert column '{col}' to datetime.")

    # 1. Email Length (Number of words in the content)
    if content_col in email_df.columns:
        email_df['email_length'] = email_df[content_col].astype(str).apply(lambda x: len(x.split()))
    else:
        print("Warning: 'content' column not found in email data.")
        email_df['email_length'] = 0  # Assign a default value

    # 2. Number of Recipients (To, CC, BCC)
    email_df['num_recipients'] = email_df[to_col].astype(str).apply(lambda x: len(x.split(';')) if isinstance(x, str) else 0) + \
                                email_df[cc_col].astype(str).apply(lambda x: len(x.split(';')) if isinstance(x, str) else 0) + \
                                email_df[bcc_col].astype(str).apply(lambda x: len(x.split(';')) if isinstance(x, str) else 0)

    # 3.  Email Sent Outside Working Hours
    if date_col in email_df.columns:
        email_df['email_hour'] = email_df[date_col].dt.hour
        email_df['off_hours_email'] = ((email_df['email_hour'] < 8) | (email_df['email_hour'] > 18)).astype(int)
        email_df.drop('email_hour', axis=1, inplace=True)
    else:
        print("Warning: 'date' column not found in email data.")
        email_df['off_hours_email'] = 0

    return email_df


def create_device_features(device_df):
    if device_df.empty:
        print("Warning: Device DataFrame is empty. Returning original DataFrame.")
        return device_df

     # ***CORRECTED COLUMN NAMES***
    date_col = 'date'
    user_col = 'user'
    pc_col = 'pc'
    activity_col = 'activity'

    # Ensure time columns are datetime objects
    for col in device_df.columns:
        if 'date' in col.lower():  # try to identify date/time column
            try:
                device_df[col] = pd.to_datetime(device_df[col], errors='coerce') # Convert to datetime, handle errors
            except:
                print(f"Warning: Could not convert column '{col}' to datetime.")

    # 1. Connection Duration (Requires separate Connect and Disconnect events)
    # More complex time series analysis needed.  Skipping for now.

    # 2.  Connection Time Outside of Normal Hours
    if date_col in device_df.columns:
        device_df['connection_hour'] = device_df[date_col].dt.hour
        device_df['off_hours_connection'] = ((device_df['connection_hour'] < 8) | (device_df['connection_hour'] > 18)).astype(int)
        device_df.drop('connection_hour', axis=1, inplace=True)
    else:
        print("Warning: 'date' column not found in device data.")
        device_df['off_hours_connection'] = 0

    return device_df


def create_file_features(file_df):
    if file_df.empty:
        print("Warning: File DataFrame is empty. Returning original DataFrame.")
        return file_df

    # ***CORRECTED COLUMN NAMES***
    date_col = 'date'
    user_col = 'user'
    pc_col = 'pc'
    filename_col = 'filename'
    content_col = 'content'

    # Ensure timestamp column is datetime objects
    for col in file_df.columns:
        if 'date' in col.lower():  # try to identify date/time column
            try:
                file_df[col] = pd.to_datetime(file_df[col], errors='coerce') # Convert to datetime, handle errors
            except:
                print(f"Warning: Could not convert column '{col}' to datetime.")

    # 1. File Size (Length of the content string)
    if content_col in file_df.columns:
        file_df['file_size'] = file_df[content_col].astype(str).apply(lambda x: len(x))
    else:
        print("Warning: 'content' column not found in file data.")
        file_df['file_size'] = 0

    # 2. Access Time (When was the file accessed?)
    if date_col in file_df.columns:
        file_df['file_access_hour'] = file_df[date_col].dt.hour
        file_df['off_hours_file_access'] = ((file_df['file_access_hour'] < 8) | (file_df[date_col].dt.hour > 18)).astype(int)
        file_df.drop('file_access_hour', axis=1, inplace=True)
    else:
        print("Warning: 'date' column not found in file data.")
        file_df['off_hours_file_access'] = 0

    return file_df


def create_psychometric_features(psychometric_df):
    if psychometric_df.empty:
        print("Warning: Psychometric DataFrame is empty. Returning original DataFrame.")
        return psychometric_df

    # ***CORRECTED COLUMN NAMES***
    o_col = 'O'
    c_col = 'C'
    e_col = 'E'
    a_col = 'A'
    n_col = 'N'


    # No new features are engineered here, as the existing columns are used directly.
    # You might want to create ratios or combinations of these scores later.

    return psychometric_df

# Apply the feature engineering functions
logon_df = create_logon_features(logon_df)
email_df = create_email_features(email_df)
device_df = create_device_features(device_df)
file_df = create_file_features(file_df)
psychometric_df = create_psychometric_features(psychometric_df)


# --- Data Preprocessing (after Feature Engineering) ---
# Assuming you have DataFrames: logon_df, email_df, device_df, file_df, psychometric_df

# 1. Handle Missing Values (example: imputation with the mean)
# IMPORTANT: Do this *before* encoding categorical features.
logon_df.fillna(logon_df.mean(numeric_only=True), inplace=True) #numeric_only=True to avoid error with non-numeric columns
email_df.fillna(email_df.mean(numeric_only=True), inplace=True)
device_df.fillna(device_df.mean(numeric_only=True), inplace=True)
file_df.fillna(file_df.mean(numeric_only=True), inplace=True)
psychometric_df.fillna(psychometric_df.mean(numeric_only=True), inplace=True)

# For categorical columns with NaNs, fill with the most frequent value:
for df in [logon_df, email_df, device_df, file_df, psychometric_df]: #Include psychometric_df
    for col in df.select_dtypes(include='object').columns: # Select object columns
        df[col] = df[col].fillna(df[col].mode()[0]) # Fill NaN with the most frequent value

# 2. Encode Categorical Features (example: Label Encoding)
label_encoder = LabelEncoder()

if 'logon_time_category' in logon_df.columns:
    logon_df['logon_time_category'] = label_encoder.fit_transform(logon_df['logon_time_category'])

# Apply Label Encoding to other categorical features in other DataFrames as needed.

# 3. Scale Numerical Features (example: StandardScaler)
scaler = StandardScaler()

# Identify numerical columns in each DataFrame
numerical_cols_logon = logon_df.select_dtypes(include=np.number).columns
numerical_cols_email = email_df.select_dtypes(include=np.number).columns
numerical_cols_device = device_df.select_dtypes(include=np.number).columns
numerical_cols_file = file_df.select_dtypes(include=np.number).columns
numerical_cols_psychometric = psychometric_df.select_dtypes(include=np.number).columns

# Debug: Print the numerical columns found
print("\nNumerical columns in Logon Data:", numerical_cols_logon)
print("Numerical columns in Email Data:", numerical_cols_email)
print("Numerical columns in Device Data:", numerical_cols_device)
print("Numerical columns in File Data:", numerical_cols_file)
print("Numerical columns in Psychometric Data:", numerical_cols_psychometric)


# Apply scaling to each DataFrame
# Add a check to ensure there are numerical columns before scaling
if len(numerical_cols_logon) > 0:
    logon_df[numerical_cols_logon] = scaler.fit_transform(logon_df[numerical_cols_logon])
if len(numerical_cols_email) > 0:
    email_df[numerical_cols_email] = scaler.fit_transform(email_df[numerical_cols_email])
if len(numerical_cols_device) > 0:
    device_df[numerical_cols_device] = scaler.fit_transform(device_df[numerical_cols_device])
if len(numerical_cols_file) > 0:
    file_df[numerical_cols_file] = scaler.fit_transform(file_df[numerical_cols_file])
if len(numerical_cols_psychometric) > 0:
    psychometric_df[numerical_cols_psychometric] = scaler.fit_transform(psychometric_df[numerical_cols_psychometric])

print("\nLogon Data after Preprocessing:")
print(logon_df.head())
print("\nEmail Data after Preprocessing:")
print(email_df.head())
print("\nDevice Data after Preprocessing:")
print(device_df.head())
print("\nFile Data after Preprocessing:")
print(file_df.head())
print("\nPsychometric Data after Preprocessing:")
print(psychometric_df.head())

# --- Data Aggregation and Combination ---

# 1. Aggregate Data (Example: Count events per user)
# You'll need to customize this based on the features you engineered.

logon_counts = logon_df.groupby('user').size().reset_index(name='logon_count')
email_counts = email_df.groupby('user').size().reset_index(name='email_count')
device_counts = device_df.groupby('user').size().reset_index(name='device_count')
file_counts = file_df.groupby('user').size().reset_index(name='file_count')

# Example: Aggregate numerical features (mean, sum, etc.)
logon_agg = logon_df.groupby('user')[['off_hours_logon']].mean().reset_index()  #Only take the numerical columns
email_agg = email_df.groupby('user')[['email_length', 'off_hours_email']].mean().reset_index()
device_agg = device_df.groupby('user')[['off_hours_connection']].mean().reset_index() #Only take the numerical columns
file_agg = file_df.groupby('user')[['file_size', 'off_hours_file_access']].mean().reset_index()


# 2. Merge DataFrames
# Start with psychometric data as the base, assuming every user is represented there.
# If not, choose another DataFrame or create a user list separately.

# Rename 'user_id' to 'user' to match the other DataFrames
psychometric_df = psychometric_df.rename(columns={'user_id': 'user'})

combined_df = psychometric_df.copy() # Start with a copy to avoid modifying the original

# Merge in the aggregated data, using a left merge to preserve all users from psychometric_df
combined_df = pd.merge(combined_df, logon_counts, on='user', how='left')
combined_df = pd.merge(combined_df, email_counts, on='user', how='left')
combined_df = pd.merge(combined_df, device_counts, on='user', how='left')
combined_df = pd.merge(combined_df, file_counts, on='user', how='left')

combined_df = pd.merge(combined_df, logon_agg, on='user', how='left')
combined_df = pd.merge(combined_df, email_agg, on='user', how='left')
combined_df = pd.merge(combined_df, device_agg, on='user', how='left')
combined_df = pd.merge(combined_df, file_agg, on='user', how='left')

# Fill NaN values that result from the merges (users might not have data in all tables)
combined_df.fillna(0, inplace=True)  # Or use a different strategy, like imputing with the mean

print("\nCombined DataFrame:")
print(combined_df.head())


# --- Anomaly Detection ---

# 1. Prepare Data for Modeling
# Drop the 'user' and 'employee_name' columns as they're not used for anomaly detection.  Keep other features.
if 'user' in combined_df.columns:
    X = combined_df.drop(['user', 'employee_name'], axis=1, errors='ignore')  # Ignore if 'employee_name' is not present
else:
    X = combined_df.drop('employee_name', axis=1, errors='ignore') # If no user column, use all data

# 2. Train the Isolation Forest Model
# Adjust parameters like n_estimators (number of trees) and contamination (expected proportion of outliers)
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X)


# 3. Predict Anomalies
# -1 indicates an anomaly, 1 indicates normal data
y_pred = model.predict(X)

# 4. Add Anomaly Scores and Predictions to the DataFrame
combined_df['anomaly_score'] = model.decision_function(X)  # Raw anomaly score
combined_df['is_anomaly'] = y_pred  # -1 for anomaly, 1 for normal

print("\nCombined DataFrame with Anomaly Scores:")
print(combined_df.head())

# --- Analyze Results ---

# 1. Count Anomalies
num_anomalies = combined_df[combined_df['is_anomaly'] == -1].shape[0]
print(f"\nNumber of Anomalies Detected: {num_anomalies}")

# 2. Investigate Anomalies (Example)
anomalies = combined_df[combined_df['is_anomaly'] == -1]
print("\nAnomalous Users:")
print(anomalies)


# --- Visualization (Optional) ---
# This requires choosing two features to plot