import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

sns.set_style('darkgrid')


df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V3 med Haversine.csv', delimiter=';',
                  dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%d/%m/%Y %H:%M')

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V4 med Haversine.csv',
                  delimiter=';')
df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%Y-%m-%d %H:%M:%S')

print(df1.head())
print(df2.head())


def FindExtremeDates(df):
    df_copy = df.copy()  # make a copy so that the original dataframe is not altered
    df_copy['Date'] = df_copy['Datetime'].dt.date.astype('str').str[-5:]
    df_copy.dropna(subset=['Datetime'], inplace=True)
    # set an arbitrary year to make it a DateTime object, but it is only m and d that we are interested in
    min_date = '2000-' + df_copy['Date'].min()
    max_date = '2000-' + df_copy['Date'].max()
    return min_date, max_date


def DateActivity(df):
    min_date, max_date = FindExtremeDates(df)
    date_range = pd.date_range(start=min_date, end=max_date).astype('str').str[-5:].tolist()
    activity = np.zeros(len(date_range))
    freq = np.zeros(len(date_range))
    df_copy = df.copy()  # make a copy so that the original dataframe is not altered
    df_copy['Date'] = df_copy['Datetime'].dt.date.astype('str').str[-5:]
    df_copy.dropna(subset=['Datetime'], inplace=True)
    df_copy.reset_index(inplace=True, drop=True)

    i = 0
    while i < len(df_copy):
        activity[date_range.index(df_copy.at[i, 'Date'])] += df_copy.at[i, 'Haversine']
        freq[date_range.index(df_copy.at[i, 'Date'])] += 1
        i += 1
    for j in range(len(activity)):
        activity[j] = activity[j] / freq[j]
    date_range = pd.to_datetime(date_range, format='%m-%d')
    return date_range, activity


dates1, activity1 = DateActivity(df1)
dates2, activity2 = DateActivity(df2)


# Plot of activity per date in Tingvoll
fig1, ax1 = plt.subplots(figsize=(16, 8))
plt.bar(x=dates1, height=activity1)
ax1.set_xlabel(' ', fontsize=1)
ax1.set_ylabel('Velocity, m/hr', fontsize=35, labelpad=30)
ax1.set_title('Mean activity per date in Tingvoll', fontsize=40, pad=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
date_form = DateFormatter("%d.%m")
ax1.xaxis.set_major_formatter(date_form)


# Plot of activity per date in Fosen
fig2, ax2 = plt.subplots(figsize=(16, 8))
plt.bar(x=dates2, height=activity2)
ax2.set_xlabel(' ', fontsize=1)
ax2.set_ylabel('Velocity, m/hr', fontsize=35, labelpad=30)
ax2.set_title('Mean activity per date in Fosen', fontsize=40, pad=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
ax2.xaxis.set_major_formatter(date_form)

plt.show()
