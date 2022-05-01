import pandas as pd
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
import time

lock = Lock()

time0 = time.time()

df1 = pd.read_csv("/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V9 med temp.csv", delimiter=';',
                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y %m %d %H:%M:%S')
df1['Altitude'] = None

print(df1)


def Elevation(lat, long, i):
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
        if (i % 10) == 0:
            print("Reached number: ", i, '--------')

        if pd.isnull(df.at[i, 'Datetime']):
            i += 1  # start the next check on the next data set in the second point
            continue

        try:
            global lock
            with lock:
                df.at[i, 'Altitude'] = Elevation(df.at[i, 'Lat'], df.at[i, 'Lon'], i)

        except Exception as e:
            exceptions += 1
            print(e)
        if i == 50:
            print(df.head(50))
        i += 1
    print('there were', exceptions, 'exceptions')


def GetValues(df):
    interval = int((len(df)) / 3)
    start = 0
    first = interval
    second = 2*interval
    end = len(df)
    return start, first, second, end


start1, first1, second1, end1 = GetValues(df1)

df_test1 = df1.iloc[start1:first1, :]
df_test2 = df1.iloc[first1:second1, :]
df_test3 = df1.iloc[second1:end1, :]

threads = [
    Thread(target=InsertAltitude, args=(df_test1, start1, first1)),
    Thread(target=InsertAltitude, args=(df_test2, first1, second1)),
    Thread(target=InsertAltitude, args=(df_test3, second1, end1))
]

for t in threads:
    t.start()

for j in threads:
    j.join()

df1 = pd.concat([df_test1, df_test2, df_test3])
df1.to_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V10 altitude test.csv', index=False, sep=';')
print('the program took ', time.time() - time0, 'seconds')

