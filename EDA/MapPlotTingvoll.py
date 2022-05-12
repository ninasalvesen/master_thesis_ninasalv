import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V3 med Haversine.csv', delimiter=';',
                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H:%M')

print(df.head())

lat = []
long = []
time = []
dataset = 2
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


# Choose which home range to look at by changing home_range and fig1
home_range = 'koksvik'
figure_koksvik = [8.1230, 8.4114, 62.8333, 62.9393]
figure_torjul = [8.0576, 8.3460, 62.9087, 63.0232]

figure = plt.imread(f"/Users/ninasalvesen/Documents/Sauedata/Bilder/figure_{home_range}.png")
fig1 = figure_koksvik

fig, ax = plt.subplots()
ax.scatter(long, lat, zorder=1, alpha=0.6, c="b", s=1)
ax.imshow(figure, zorder=0, extent=fig1, aspect="auto")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid("True")
plt.show()
