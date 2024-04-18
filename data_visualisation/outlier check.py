import os
import pandas as pd

folder = 'dataCSV/Jump'

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder) if file.endswith('.csv')]
print(len(csv_files))
# Initialize a list to store the number of rows in each DataFrame
num_rows_list = []

# Iterate through each CSV file
for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(os.path.join(folder, file))

    print(f"{file}: {df.shape}")
    # Get the number of rows in the DataFrame
    num_rows = df.shape[0]
    # Append the number of rows to the list
    num_rows_list.append(num_rows)

# Calculate mean and standard deviation of the number of rows
mean_num_rows = sum(num_rows_list) / len(num_rows_list)
std_num_rows = (sum((x - mean_num_rows) ** 2 for x in num_rows_list) / len(num_rows_list)) ** 0.5

# Define a threshold for outliers (for example, consider rows outside mean ± 2 * std as outliers)
lower_threshold = mean_num_rows - 1 * std_num_rows
upper_threshold = mean_num_rows + 1 * std_num_rows

# Identify outliers
outliers = [(file, num_rows) for file, num_rows in zip(csv_files, num_rows_list) if num_rows < lower_threshold or num_rows > upper_threshold]

# Print outliers
for outlier in outliers:
    print(f"Outlier: {outlier[0]}, Number of Rows: {outlier[1]}")

print(f"Mean: {mean_num_rows}, Standard Deviation: {std_num_rows}")
print(len(csv_files))
counter = 0
# Ask user if they want to delete specific files
for outlier in outliers:
    user_input = input(f"Do you want to delete the outlier file '{outlier[0]}', Number of Rows: {outlier[1]}? (Press 'space' to delete, any other key to keep): ")
    if user_input == "x":
        counter += 1
        os.remove(os.path.join(folder, outlier[0]))
        print(f"File '{outlier[0]}' deleted, {len(csv_files)}")
        print(f"Number of files deleted: {counter}")
    else:
        print(f"File '{outlier[0]}' kept, {len(csv_files)}")
