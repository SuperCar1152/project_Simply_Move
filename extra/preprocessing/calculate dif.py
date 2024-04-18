import pandas as pd

# Read the CSV file
df = pd.read_csv('logfile_D4-22-CD-00-49-AA.csv')

# Drop first row
df = df.drop(df.index[0]).reset_index(drop=True)

# Calculate differences in SampleTimeFine column
df['TimeDifference'] = df['SampleTimeFine'].diff()

#Optionally, handle NaN values or drop them
df['TimeDifference'] = df['TimeDifference'].fillna(0)  # Replace NaN with 0
df = df.dropna()  # Drop rows with NaN values

# Output the DataFrame with differences
print(df)
