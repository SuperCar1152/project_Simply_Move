import os
import pandas as pd

folder = 'dataCSV/Clap'

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder) if file.endswith('.csv')]
print(len(csv_files))
# Iterate through each CSV file
for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(os.path.join(folder, file))
    print(f"{file}: {df.shape}")
