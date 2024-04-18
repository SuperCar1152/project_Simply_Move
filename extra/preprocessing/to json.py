import os
import pandas as pd
import matplotlib.pyplot as plt

folder = 'dataCSV/leg/OtherMove_20240411_163549.csv'

df = pd.read_csv(folder)

json_data = df.to_json()

with open('output.json', 'w') as f:
    f.write(json_data)
