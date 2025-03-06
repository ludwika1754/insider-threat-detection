import pandas as pd

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

# Basic data exploration (do this for each DataFrame)
print("\nLogon Data:")
print(logon_df.head())
print(logon_df.info())
print(logon_df.describe())

print("\nEmail Data:")
print(email_df.head())
print(email_df.info())
print(email_df.describe())

print("\nDevice Data:")
print(device_df.head())
print(device_df.info())
print(device_df.describe())

print("\nFile Data:")
print(file_df.head())
print(file_df.info())
print(file_df.describe())

print("\nPsychometric Data:")
print(psychometric_df.head())
print(psychometric_df.info())
print(psychometric_df.describe())

# Check for missing values
print("\nMissing Values (Logon):")
print(logon_df.isnull().sum())

print("\nMissing Values (Email):")
print(email_df.isnull().sum())

print("\nMissing Values (Device):")
print(device_df.isnull().sum())

print("\nMissing Values (File):")
print(file_df.isnull().sum())

print("\nMissing Values (Psychometric):")
print(psychometric_df.isnull().sum())