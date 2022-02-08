import pandas as pd
import numpy as np

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll uten timeclean.csv', delimiter=';',
                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen uten timeclean.csv', delimiter=';')

df1['Datetime'] = df1['Date'] + " " + df1['Time']
df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')

df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%Y-%m-%d %H:%M:%S')


def timeclean(df, delta):
    month_end = [30, 31]
    count = 0
    temp = 1     # data set number, skips the first row so must be initialized at 1
    i = 1        # have to check the first and two last lines for faulty points manually
    while i < (len(df)-2):
        if pd.isnull(df.at[i + 1, 'Datetime']):
            temp += 1
            i += 2  # start the next check on the next data set
            continue

        year_diff = df.at[i+1, 'Datetime'].year - df.at[i, 'Datetime'].year
        month_diff = df.at[i+1, 'Datetime'].month - df.at[i, 'Datetime'].month
        day_diff = df.at[i+1, 'Datetime'].day - df.at[i, 'Datetime'].day

        if year_diff != 0:
            count += 1
            print('year', i)
            df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(year=df.at[i, 'Datetime'].year)
            continue  # check again that it works before moving on

        if abs(month_diff) > 1:
            count += 1
            print('month', i)
            df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(month=df.at[i, 'Datetime'].month)
            continue

        if month_diff == 1 and df['Datetime'][i].day not in month_end:
            if df['Datetime'][i+2].month == df['Datetime'][i].month:
                count += 1
                print('month2', i)
                df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(month=df.at[i, 'Datetime'].month)
            else:
                i += 1
            continue

        if month_diff == -1:
            count += 1
            print('month3', i)
            df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(month=df.at[i, 'Datetime'].month)
            continue

        if abs(day_diff) > 1 and df['Datetime'][i].day not in month_end:
            count += 1
            print('day', i)
            df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(day=df.at[i, 'Datetime'].day)
            continue

        if day_diff == 1 and df['Datetime'][i].time().hour < 22:
            if df['Datetime'][i + 2].day == df['Datetime'][i].day:
                count += 1
                print('day2', i)
                df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(day=df.at[i, 'Datetime'].day)
            else:
                i += 1
            continue

        if day_diff == -1:
            count += 1
            print('day3', i)
            df.at[i + 1, 'Datetime'] = df.at[i + 1, 'Datetime'].replace(day=df.at[i, 'Datetime'].day)
            continue

        time_diff = pd.Timedelta(np.abs(df['Datetime'][i + 1] - df['Datetime'][i])).seconds / (60 * 60)
        if time_diff > delta:     # delta enables the user to choose the time difference limit in hours
            if pd.isnull(df.at[i + 2, 'Datetime']) == False and pd.Timedelta(np.abs(df['Datetime'][i + 2] - df['Datetime'][i])).seconds / (60 * 60) < delta:
                count += 1
                mid_diff = pd.Timedelta(np.abs(df.at[i+2, 'Datetime'] - df.at[i, 'Datetime'])).seconds / 2
                df.at[i + 1, 'Datetime'] = df.at[i, 'Datetime'] + pd.Timedelta(hours=mid_diff/(60*60))
                print('time', i)

        if (i % 10000) == 0:
            # check progress against number of iterations
            print("Reached number: ", i)

        i += 1
    return df, count


#df1, count = timeclean(df1, 2)
#df1 = df1.drop(columns=['Date', 'Time'])  # deleting the (uncleaned) columns that we no longer need
#df1.to_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll.csv', index=False, sep=';')

df2, count = timeclean(df2, 4)
#df2 = df2.drop(columns=['Date.Time'])
#df2.to_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen.csv', index=False, sep=';')

print(count)
