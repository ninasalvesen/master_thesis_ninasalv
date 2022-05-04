import pandas as pd
import numpy as np

df1 = pd.read_csv("C:\\Users\\ovsal\\Documents\\Nina Master\\Samlet data Fosen V9 info_features med temp.csv", delimiter=';',
                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y %m %d %H:%M:%S')
df1['angle'] = 0.0

print(df1)


def angle(df):
    count = 0
    i = 2
    while i < len(df)-1:
        if pd.isnull(df.at[i + 1, 'Datetime']):
            count += 1
            print(count)
            i += 3
            continue
        vec1 = [df.at[i - 1, 'Lat'] - df.at[i, 'Lat'], df.at[i - 1, 'Lon'] - df.at[i, 'Lon']]
        vec2 = [df.at[i + 1, 'Lat'] - df.at[i, 'Lat'], df.at[i + 1, 'Lon'] - df.at[i, 'Lon']]
        len1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
        len2 = np.sqrt(vec2[0] ** 2 + vec2[1] ** 2)
        dot = np.dot(vec1, vec2)
        
        # some vectors will be very small, check for no change (cannot divide by zero)
        if len1*len2 == 0 and dot == 0:
            df.at[i, 'angle'] = 0
        else:
            df.at[i, 'angle'] = 180 - (np.arccos(round(dot/(len1*len2), 7)) * 180 / np.pi)
            # take 180 - angle to find directional change, where big change = big number
        i += 1

angle(df1)
print(df1)
