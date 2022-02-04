import pandas as pd

df = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll.csv', delimiter=';',
                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

df['Datetime'] = df['Date'] + " " + df['Time']
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
print(df.head())

month_end = [30, 31]
temp = 1
i = 1
while i < (len(df)-2):
    if pd.isnull(df.at[i + 1, 'Datetime']):
        temp += 1
        i += 2  # start the next check on the next data set
        continue

    year_diff = df.at[i+1, 'Datetime'].year - df.at[i, 'Datetime'].year
    month_diff = df.at[i+1, 'Datetime'].month - df.at[i, 'Datetime'].month
    day_diff = df.at[i+1, 'Datetime'].day - df.at[i, 'Datetime'].day

    if year_diff != 0:
        df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(year=df.at[i, 'Datetime'].year)
        continue  # check again that it works before moving on

    if abs(month_diff) > 1:
        df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(month=df.at[i, 'Datetime'].month)
        continue

    if month_diff == 1 and df['Datetime'][i].day not in month_end:
        if df['Datetime'][i+2].month == df['Datetime'][i].month:
            df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(month=df.at[i, 'Datetime'].month)
        else:
            i += 1
        continue

    if month_diff == -1:
        df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(month=df.at[i, 'Datetime'].month)
        continue

    if abs(day_diff) > 1 and df['Datetime'][i].day not in month_end:
        df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(day=df.at[i, 'Datetime'].day)
        continue

    if day_diff == 1 and df['Datetime'][i].time().hour < 22:
        if df['Datetime'][i + 2].day == df['Datetime'][i].day:
            df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(day=df.at[i, 'Datetime'].day)
        else:
            i += 1
        continue

    if day_diff == -1:
        df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(day=df.at[i, 'Datetime'].day)
        continue

    time_diff = pd.Timedelta(abs(df['Datetime'][i + 1] - df['Datetime'][i])).seconds / (60 * 60)
    if time_diff > 2:
        if pd.Timedelta(abs(df['Datetime'][i + 2] - df['Datetime'][i])).seconds / (60 * 60) < 2:
            mid_diff = pd.Timedelta(abs(df.at[i+2, 'Datetime'] - df.at[i, 'Datetime'])).seconds / 2
            df.at[i + 1, 'Datetime'] = df.at[i, 'Datetime'] + pd.Timedelta(hours=mid_diff/(60*60))

    if (i % 10000) == 0:
        # check progress against number of iterations
        print("Reached number: ", i)

    i += 1

df = df.drop(columns=['Date', 'Time'])  # deleting the (uncleaned) columns that we no longer need
#df.to_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll.csv', index=False, sep=';')