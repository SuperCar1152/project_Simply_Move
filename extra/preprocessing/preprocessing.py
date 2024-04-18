import pandas as pd

df = pd.read_csv("TimeStampFromTimeDiffPlace.csv")

df = df.sort_values(by="Timestamp")

df.to_csv("PreprocessedCorrectTimeStamp.csv", index=False)