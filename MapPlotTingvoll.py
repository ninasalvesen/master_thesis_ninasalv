import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll.csv", delimiter=';',
                 dtype={"Initial start": "str", "Start": "str", "Stop": "str"})
print(df.head())

temp = 0
dataset = 6

lat = []
long = []
for i in range(len(df)):
    if "new_sheep" in df.iloc[i][0]:
        temp += 1
        continue
    if temp == dataset:
        lat.append(df['Lat'][i])
        long.append(df['Long'][i])
    if temp > dataset:
        break


extreme_values = [min(long), max(long), min(lat), max(lat)]
print("extreme values: ", extreme_values)
print('Len: ', len(lat))


# Choose which home range to look at by changing home_range and fig1
home_range = "koksvik"
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