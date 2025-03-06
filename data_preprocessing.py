import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# Example: Feature Engineering for the 'file' data

def feature_engineer_file(df, time_window='D'):  # D for day, H for hour, etc.
    """
    Feature engineering function for file activity data.
    """

    # 1. Convert 'date' to datetime objects
    df['date'] = pd.to_datetime(df['date'])

    # 2. Time-based aggregation (group by user and time window)
    df['time_window'] = df['date'].dt.to_period(time_window)
    grouped = df.groupby(['user', 'time_window'])

    # 3. Calculate basic counts
    feature_df = grouped.agg(
        file_count = ('filename','count'),
        unique_file_count = ('filename', 'nunique')
    ).reset_index()

    # 4. Extract file extensions
    df['file_extension'] = df['filename'].str.extract(r'(\.[^.]+)$')

    # One Hot Encode file_extension
    file_extension_encoded = pd.get_dummies(df['file_extension'], prefix='file_type', dummy_na=False)
    df = pd.concat([df, file_extension_encoded], axis=1)

    feature_df_extension = df.groupby(['user', 'time_window'])[[col for col in df.columns if 'file_type' in col]].sum().reset_index()

    feature_df = pd.merge(feature_df, feature_df_extension, on=['user','time_window'])

    # 5. Content Length feature
    df['content_length'] = df['content'].str.len()
    feature_df_content = df.groupby(['user','time_window']).agg(
        content_length_avg = ('content_length','mean'),
        content_length_sum = ('content_length', 'sum')
    ).reset_index()

    feature_df = pd.merge(feature_df, feature_df_content, on = ['user', 'time_window'])

    #6 . File Access Rarity

    file_counts = df['filename'].value_counts().reset_index()
    file_counts.columns = ['filename', 'rarity_count']
    df = pd.merge(df, file_counts, on='filename', how='left')

    feature_df_rarity = df.groupby(['user', 'time_window']).agg(
        file_access_rarity_avg = ('rarity_count','mean'),
        file_access_rarity_sum = ('rarity_count', 'sum')
    ).reset_index()

    feature_df = pd.merge(feature_df, feature_df_rarity, on = ['user', 'time_window'])

    return feature_df #Include time_window and user columns

# Example usage:
file_path = r"C:\Users\ludwi\OneDrive\Desktop\insider-threat-project\f.xlsx"  # Store the path
file_df = pd.read_excel(file_path)  # Read the Excel file into a DataFrame
file_features = feature_engineer_file(file_df.copy()) # Make a copy to avoid modifying the original

print(file_features.head())


def feature_engineer_device(df, time_window='D'):
    """
    Feature engineering function for device activity data.
    """

    # 1. Convert 'date' to datetime objects
    df['date'] = pd.to_datetime(df['date'])

    # 2. Time-based aggregation (group by user and time window)
    df['time_window'] = df['date'].dt.to_period(time_window)
    grouped = df.groupby(['user', 'time_window'])

    # 3. Calculate basic counts
    feature_df = grouped.agg(
        device_activity_count = ('activity','count')
    ).reset_index()

    # 4. One-hot encode device activities
    device_activity_encoded = pd.get_dummies(df['activity'], prefix='device_activity', dummy_na=False)
    df = pd.concat([df, device_activity_encoded], axis=1)

    # Group by user and time window to sum the one-hot encoded features
    feature_df_activity = df.groupby(['user', 'time_window'])[[col for col in df.columns if 'device_activity' in col]].sum().reset_index()

    # Merge the counts and one-hot encoded features
    feature_df = pd.merge(feature_df, feature_df_activity, on=['user', 'time_window'])

    return feature_df

#Example Usage
device_path = r"C:\Users\ludwi\OneDrive\Desktop\insider-threat-project\d.xlsx"
device_df = pd.read_excel(device_path)
device_features = feature_engineer_device(device_df.copy())

print(device_features.head())

def feature_engineer_email(df, time_window='D'):
    """
    Feature engineering function for email activity data.
    """

    # 1. Convert 'date' to datetime objects
    df['date'] = pd.to_datetime(df['date'])

    # 2. Time-based aggregation (group by user and time window)
    df['time_window'] = df['date'].dt.to_period(time_window)
    grouped = df.groupby(['user', 'time_window'])

    # 3. Basic email counts
    feature_df = grouped.agg(
        email_sent_count = ('to', 'count'),  # Assuming 'to' indicates sent emails
        unique_recipients_count = ('to', 'nunique'),
        total_email_size = ('size', 'sum'),
        attachment_count = ('attachments', 'count'),
    ).reset_index()

    # 4. Email to self feature
    df['email_to_self'] = df['to'] == df['user']
    email_to_self_count = df.groupby(['user', 'time_window'])['email_to_self'].sum().reset_index()
    feature_df = pd.merge(feature_df, email_to_self_count, on=['user', 'time_window'])

    # External recipient count (you'll need to define what an external email looks like)
    def is_external_email(email):
        #Example implemenation
        return not email.endswith("@yourcompany.com")  #Replace with your company domain

    df['is_external'] = df['to'].apply(is_external_email)
    external_recipients_count = df.groupby(['user', 'time_window'])['is_external'].sum().reset_index()
    feature_df = pd.merge(feature_df, external_recipients_count, on=['user', 'time_window'])

    return feature_df

#Example Usage
email_path = r"C:\Users\ludwi\OneDrive\Desktop\insider-threat-project\e.xlsx"
email_df = pd.read_excel(email_path)
email_features = feature_engineer_email(email_df.copy())

print(email_features.head())

def feature_engineer_logon(df, time_window='D'):
    """
    Feature engineering function for logon activity data.
    """

    # 1. Convert 'date' to datetime objects
    df['date'] = pd.to_datetime(df['date'])

    # 2. Time-based aggregation (group by user and time window)
    df['time_window'] = df['date'].dt.to_period(time_window)
    grouped = df.groupby(['user', 'time_window'])

    # 3. Basic logon counts
    feature_df = grouped.agg(
        logon_count = ('activity', 'count')
    ).reset_index()

    # 4. One-hot encode logon activities
    logon_activity_encoded = pd.get_dummies(df['activity'], prefix='logon_activity', dummy_na=False)
    df = pd.concat([df, logon_activity_encoded], axis=1)

    # Group by user and time window to sum the one-hot encoded features
    feature_df_activity = df.groupby(['user', 'time_window'])[[col for col in df.columns if 'logon_activity' in col]].sum().reset_index()

    # Merge the counts and one-hot encoded features
    feature_df = pd.merge(feature_df, feature_df_activity, on=['user', 'time_window'])

    return feature_df

#Example Usage
logon_path = r"C:\Users\ludwi\OneDrive\Desktop\insider-threat-project\l.xlsx"
logon_df = pd.read_excel(logon_path)
logon_features = feature_engineer_logon(logon_df.copy())

print(logon_features.head())