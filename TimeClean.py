import numpy as np
import pandas as pd

df = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll.csv', delimiter=';',
                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})
df['Datetime'] = df['Date'] + " " + df['Time']
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y %m %d %H:%M:%S')

print(df['Datetime'].dtype)

print(df['Datetime'].dt.date[1]-df['Datetime'].dt.date[210])
print(df['Datetime'].dt.time[1])
print(df['Datetime'].dt.time[210])

count = 0 # telle antall ganger det er feil i datotid

"""
for i in range(len(df)):
    if "new_sheep" in df.iloc[i][0]:
    # sette inn noe mekanikk her så man ikke sammenlikner første og siste dato og øker counten på det
        continue
    year = df['Datetime'].dt.year[i]
    month = df['Datetime'].dt.month[i]
    day = df['Datetime'].dt.day[i]
    hour = df['Datetime'].dt.hour[i]
    minute = df['Datetime'].dt.minute[i]
    second = df['Datetime'].dt.second[i]
"""