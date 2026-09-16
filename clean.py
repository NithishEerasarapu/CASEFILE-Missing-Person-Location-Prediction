import pandas as pd
from sklearn. ensemble import IsolationForest

#Load GPS dataset
df = pd.read_csv("data/gps_data.csv")

print("Dataset loaded successfully!")
print("Original number of records:", len(df))

# Remove duplicate records
df = df.drop_duplicates()

# Remove rows with missing values
df = df.dropna()

# Clean text columns
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].astype(str).str.strip()

# Save cleaned dataset
df.to_csv("data/cleaned_gps_data.csv")

print("Cleaning completed successfully!")
print("Cleaned dataset saved successfully!")
print("Final number of records:", len(df))