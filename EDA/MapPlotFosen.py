import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Total Fosen med angle.csv',
                 delimiter=';')
df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')

print(df1)


def FindPlot(df, sett):
    lat = []
    long = []
    time = []
    dataset = sett
    check = 0
    i = 0
    while i < len(df):
        if pd.isnull(df.at[i, 'Datetime']):
            i += 1
            check += 1

        if dataset == check:
            time.append(df.at[i, 'Datetime'])
            lat.append(df.at[i, 'Lat'])
            long.append(df.at[i, 'Lon'])
            while pd.isnull(df.at[i, 'Data set']):
                lat.append(df.at[i+1, 'Lat'])
                long.append(df.at[i+1, 'Lon'])
                i += 1
                if i == (len(df) - 1):
                    time.append(df.at[i-1, 'Datetime'])
                    break
            time.append(df.at[i-1, 'Datetime'])
            break
        i += 1

    print("Time frame for the time spent on pastures: ", time)
    extreme_values = [min(long), max(long), min(lat), max(lat)]
    print("extreme values: ", extreme_values)
    return lat, long


figurebig = plt.imread(f"/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/figure_fosen_big.png")
figbig = [10.1218, 10.6986, 63.7005, 63.9229]

lat1, long1 = FindPlot(df1, 159)
lat2, long2 = FindPlot(df1, 170)
lat3, long3 = FindPlot(df1, 235)

fig, ax = plt.subplots(figsize=(16, 16))
farm1 = ax.scatter(long1, lat1, zorder=1, alpha=0.6, c="b", s=10)
farm2 = ax.scatter(long2, lat2, zorder=1, alpha=0.6, c="r", s=10)
farm3 = ax.scatter(long3, lat3, zorder=1, alpha=0.6, c="g", s=10)
ax.imshow(figurebig, zorder=0, extent=figbig, aspect="auto")
ax.set_xlabel('Longitude', fontsize=30, labelpad=20)
ax.set_ylabel('Latitude', fontsize=30, labelpad=20)
ax.set_title('Sheep range map in Fosen', fontsize=40, pad=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.grid("True")
plt.legend(['Farm 1', 'Farm 2', 'Farm 3'], loc='lower right', prop={'size': 30}, frameon=True, markerscale=3)
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/01 Total/map_range_fosen.png", dpi=500)

plt.show()
