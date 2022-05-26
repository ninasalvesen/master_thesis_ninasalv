# Statistical analysis of the movement pattern of sheep and the occurrence of predators #
## Master thesis for Nina Salvesen ##

This is the git repository for the master thesis for Nina Salvesen, in the field of applied physics and computer science at the Norwegian University of Science and Technology. The goal of the study is to find the normal and thus abnormal behavior of different Norwegian sheep breeds on outfield pastures, with the use of data driven analysis, statistics and machine learning. Threshold values for the dynamic features indicating abnormal behavior were found. All code files are described below.

The retrieved data is from two different areas in Norway, for different herds of sheep. The two areas are Fosen in Trøndelag, and Tingvoll in Møre og Romsdal.

## DataCleaning ##
The total files used for data wrangling, that is cleaning and transforming all the raw data into a useful format.

### FosenDelete.py ###
The data collected from Fosen, Trøndelag, had several data sets included which did not contain any further information on the sheep and were therefore not particularily useful for the analysis. These sets had to be deleted.

### PointClean.py ###
Fosen had some faulty points which generated to be zeros whenever a point generation failed. New values had to be imputed inplace of the zeros.

### TimeClean.py ###
Both Tingvoll and Fosen had data were the time feature were erronous in its generation, which had to be fixed. There were a multitude of different faults that could happen to the date and time, demanding different solutions.

### TimeInterval.py ###
A uniform time interval had to be set for all sets in order to be able to compare activity across years. The dates are shown below.

| Fosen         | Start date      | End date          |
| :------------ |:--------------- | :-----            |
| 2018          | 03.06           | 29.06             |
| 2019          | 03.06           | 03.07 or 31.08    |
| 2020          | 03.06           | 05.09             |

| Tingvoll      | Start date (Koksvik/Torjul)| End date |
| :------------ |:--------------- | :-----              |
| 2012          | 09.06           | 07.09               |
| 2013          | 23.06 / 15.06   | 25.08               |
| 2014          | 05.06 / 25.06   | 10.09               |
| 2015          | 13.06 / 03.07   | 06.09               |
| 2016          | 17.06           | 22.07               |

### TingvollReduce.py ###
Tingvoll had twice the resolution of points, and in order for the data to be comparable they have to have the same time interval between each example. Therefore Tingvoll had to delete every other point if the time was less than that of Fosen.

### UpdateFormat.py ###
The retrieved data came with extraneous information not useful for the data science in this thesis. These columns had to be deleted.



## EDA - Exploratory Data Analysis ##

### ActivityPerTime.py ###
Uses the feature of velocity per hour (Haversine) to display the activity of the sheep per year, date and hour. Also make box plots to show statistical information on the activity per hour.

### eda.py ###
Plots correlation plots between features in the form of a heatmap and a scatter pairplot matrix.

### InfoGenerationFosen.py ###
Generates a csv-file with all the information on each sheep data set.

### InfoGenerationTingvoll.py ###
Generates a csv-file with all the information on each sheep data set.

### MapPlotFosen.py ###
Code to plot the trajectory of the sheep in Fosen against a map.

### MapPlotTingvoll.py ###
Code to plot the trajectory of the sheep in Tingvoll against a map.

### SizeCheck.py ###
Checks the size of each sheep data set and plot the results.

### StartEndDates.py ###
Plots the start and end dates in Fosen and Tingvoll before alteration, so this information can be used to decide what threshold dates to set.


## FeatureEngineering ##

### Altitude.py ###
Retrieves the altitude value for each latitude, longitude-pair in the data, by the use of API-calls to Kartverket and multithreading.

### Angle.py ###
Makes a feature that finds the trajectory angle between three points of movement for the sheep, in order to see when they suddenly change direction. 

### Haversine.py ###
Finds the velocity per hour for the sheep, by taking the distance travelled between two coordinates using the Haversine formula and the hour time difference between those points.

### InfoFeatureGeneration.py ###
Creates features based on the background information collected on the sheep, by connecting their id in the data to the info given for each id.

### TimeScale.py ###
Transform the datetime-feature into a sine-cosine pair to represent time as a feature that can be understood by a machine learning model.

### WeatherFeature.py ###
External data on the temperature in Fosen and Tingvoll were collected from GeoNorge, and implemented as a feature for each point.

## MachineLearning ##

### DBSCAN.py ###
Implementation of the DBSCAN machine learning clustering algorithm.

### Kmeans.py ###
Implementation of the Kmeans machine learning clustering algorithm.

### StatSignificance.py ###
Calculating the statistical significance of threshold vakues with two-sample one-tailed student t-testing, finding the percentiles, and plotting the dynamic features against the computed threshold value.


