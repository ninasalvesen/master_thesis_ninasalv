import pandas as pd
import numpy as np

# both check the haversine, and impute values where needed, and add distance per hour as a feature

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V4 after cut 2.0.csv', delimiter=';',
                  dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')
#df1['Haversine'] = 0

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V5 after cut 2.0.csv',
                  delimiter=';')
df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%Y-%m-%d %H:%M:%S')
#df2['Haversine'] = 0

print(df1.head())
print(df2.head())


def haversine(lat0, long0, lat1, long1):
    R = 6371000
    lat0 = lat0 * np.pi / 180  # in rad
    lat1 = lat1 * np.pi / 180
    long0 = long0 * np.pi / 180
    long1 = long1 * np.pi / 180
    delta_phi = lat1 - lat0
    delta_lambda = long1 - long0
    a = np.sin(delta_phi / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * (np.sin(delta_lambda / 2) ** 2)
    d = 2 * R * np.arcsin(np.sqrt(a))
    return d


def insert_speed(df):
    i = 0
    while i < (len(df)):
        if (i % 10000) == 0:
            # check progress against number of iterations
            print("Reached number: ", i)

        if pd.isnull(df.at[i, 'Datetime']):
            i += 2  # start the next check on the next data set in the second point
            continue

        dist = haversine(df.at[i, 'Lat'], df.at[i, 'Lon'], df.at[i - 1, 'Lat'], df.at[i - 1, 'Lon'])  # in meters
        time = ((df.at[i, 'Datetime'] - df.at[i - 1, 'Datetime']).total_seconds() / (60 * 60))  # in hour(s)
        speed = dist / time  # in m/hr
        df.at[i, 'Haversine'] = speed
        i += 1
    return df


def dist_check(df, dist_max):  # imputes average points where the distance is too big
    i = 0
    while i < (len(df)-2):
        if (i % 10000) == 0:
            # check progress against number of iterations
            print("Reached number: ", i)
        if pd.isnull(df.at[i, 'Datetime']):
            i += 1  # start the next check on the next data set in the second point as the first dist. point is zero
            continue

        if df.at[i + 1, 'Haversine'] > dist_max:
            if pd.isnull(df.at[i + 2, 'Datetime']):  # make sure not to impute with nans, drop the row instead
                df = df.drop(df.index[i + 1])
                df.reset_index(inplace=True, drop=True)
                continue
            dist = haversine(df.at[i + 2, 'Lat'], df.at[i + 2, 'Lon'], df.at[i, 'Lat'], df.at[i, 'Lon'])
            time = (df.at[i + 2, 'Datetime'] - df.at[i, 'Datetime']).total_seconds() / (60 * 60)
            speed = dist / time
            if speed < dist_max:
                # impute new values
                df.at[i + 1, 'Lat'] = (df.at[i, 'Lat'] + df.at[i + 2, 'Lat']) / 2
                df.at[i + 1, 'Lon'] = (df.at[i, 'Lon'] + df.at[i + 2, 'Lon']) / 2

                # update the Haversine values
                df.at[i + 1, 'Haversine'] = haversine(df.at[i + 1, 'Lat'], df.at[i + 1, 'Lon'], df.at[i, 'Lat'],
                                                      df.at[i, 'Lon']) / ((df.at[i + 1, 'Datetime'] - df.at[
                                                        i, 'Datetime']).total_seconds() / (60 * 60))
                df.at[i + 2, 'Haversine'] = haversine(df.at[i + 2, 'Lat'], df.at[i + 2, 'Lon'], df.at[i + 1, 'Lat'],
                                                      df.at[i + 1, 'Lon']) / ((df.at[i + 2, 'Datetime'] - df.at[
                                                        i + 1, 'Datetime']).total_seconds() / (60 * 60))
                continue

            elif df.at[i + 2, 'Haversine'] < dist_max:
                df.at[i + 1, 'Lat'] = (df.at[i, 'Lat'] + df.at[i + 2, 'Lat']) / 2
                df.at[i + 1, 'Lon'] = (df.at[i, 'Lon'] + df.at[i + 2, 'Lon']) / 2

                df.at[i + 1, 'Haversine'] = haversine(df.at[i + 1, 'Lat'], df.at[i + 1, 'Lon'], df.at[i, 'Lat'],
                                                      df.at[i, 'Lon']) / ((df.at[i + 1, 'Datetime'] - df.at[
                                                        i, 'Datetime']).total_seconds() / (60 * 60))
                df.at[i + 2, 'Haversine'] = haversine(df.at[i + 2, 'Lat'], df.at[i + 2, 'Lon'], df.at[i + 1, 'Lat'],
                                                      df.at[i + 1, 'Lon']) / ((df.at[i + 2, 'Datetime'] - df.at[
                                                        i + 1, 'Datetime']).total_seconds() / (60 * 60))

            elif df.at[i + 2, 'Haversine'] < dist_max:
                df = df.drop(df.index[i + 1])
                df.reset_index(inplace=True, drop=True)
                continue

            else:
                print('Manual check on', i)

        i += 1
    return df


#df1 = insert_speed(df1)
#df1 = dist_check(df1, 15000)
#df1 = insert_speed(df1)  # update velocity as final check

#df2 = insert_speed(df2)
#df2 = dist_check(df2, 15000)
#df2 = insert_speed(df2)  # update velocity as final check

#df1.to_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V4 after cut 2.0.csv', index=False, sep=';')
#df2.to_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V5 after cut 2.0.csv', index=False, sep=';')
