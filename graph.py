import os
import pandas as pd
import matplotlib.pyplot as plt

folder = 'dataCSV/Jump'

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder) if file.endswith('.csv')]

# Iterate through each CSV file
for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(os.path.join(folder, file))

    df = df.drop(columns=['Timestamp', 'Device'], errors='ignore')

    # Plot all variables in one graph
    plt.figure(figsize=(10, 6))

    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'Data Visualization - {file}')
    plt.legend()
    plt.show()
