import pandas as pd
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from threading import Thread, Lock
import time
from FeatureEngineering import Haversine, Angle

#lock = Lock()

#time0 = time.time()

#df1 = pd.read_csv("/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V10 altitude test.csv", delimiter=';',
#                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

#df2 = pd.read_csv("/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V9 info_features med temp.csv", delimiter=';',
#                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

#df2['Altitude'] = None

#inter = int((len(df2)) / 2)
#df_del1 = df2.iloc[0:inter, :]
#df_del2 = df2.iloc[inter:len(df2), :]
#df_del2.reset_index(inplace=True, drop=True)


def Elevation(lat, long):
    query = ('http://openwps.statkart.no/skwms1/wps.elevation2?request=Execute&service=WPS&version=1.0.0'
                 f'&identifier=elevation&datainputs=lat={lat};lon={long};epsg=4326')
    parsing = "{http://www.opengis.net/wps/1.0.0}"
    with urlopen(query) as f:
        tree = ET.parse(f)
        root = tree.getroot()
        return float(root.findall(f".//{parsing}Data/*")[0].text)


def InsertAltitude(df, start, stopp):
    exceptions = 0
    i = start
    while i < stopp:
        if (i % 1000) == 0:
            print("Reached number: ", i, '--------')

        if pd.isnull(df.at[i, 'Datetime']):
            i += 1  # start the next check on the next data set in the second point
            continue

        try:
            global lock
            with lock:
                df.at[i, 'Altitude'] = Elevation(df.at[i, 'Lat'], df.at[i, 'Lon'])

        except Exception as e:
            exceptions += 1
            print(e)

        if (i % 10000) == 0:
            df.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/altitude Fosen.csv', index=False, sep=';')

        i += 1
    print('there were', exceptions, 'exceptions')


def GetValues(df):
    interval = int((len(df)) / 3)
    start = 0
    first = interval
    second = 2*interval
    end = len(df)
    return start, first, second, end


def runner(df):
    start1, first1, second1, end1 = GetValues(df)
    print(start1, first1, second1, end1)

    df_test1 = df.iloc[start1:first1, :]
    df_test2 = df.iloc[first1:second1, :]
    df_test3 = df.iloc[second1:end1, :]

    threads = [
        Thread(target=InsertAltitude, args=(df_test1, start1, first1)),
        Thread(target=InsertAltitude, args=(df_test2, first1, second1)),
        Thread(target=InsertAltitude, args=(df_test3, second1, end1))
    ]

    for t in threads:
        t.start()

    for j in threads:
        j.join()

    dftot = pd.concat([df_test1, df_test2, df_test3])
    return dftot


#df_del1tot = runner(df_del1)
#df_del1tot.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Samlet data Fosen med temp del 1.csv', index=False, sep=';')
#print('the program 1 took ', time.time() - time0, 'seconds')

#df_del2tot = runner(df_del2)
#df_del2tot.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Samlet data Fosen med temp del 2.csv', index=False, sep=';')
#print('the program 2 took ', time.time() - time0, 'seconds')


def CheckWrongAltitudes(df):
    i = 0
    while i < len(df):
        if (i % 10000) == 0:
            print("Reached number: ", i, '--------')

        if pd.isnull(df.at[i, 'Datetime']):
            i += 1  # start the next check on the next data set in the second point
            continue

        if df.at[i, 'Altitude'] < 0:
            df = df.drop(df.index[i])
            df.reset_index(inplace=True, drop=True)
            continue

        i += 1

    Haversine.insert_speed(df)
    Angle.angle(df)
    return df


df_test = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/Total begge med feil altitude.csv', sep=';',
                      dtype={"Initial start": "str", "Start": "str", "Stop": "str"})
df_test['Datetime'] = pd.to_datetime(df_test['Datetime'], format='%Y-%m-%d %H:%M:%S')

dfny = CheckWrongAltitudes(df_test)
#dfny.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Samlet data begge test.csv', index=False, sep=';')


