# Statistical analysis of the movement pattern of sheep and the occurrence of predators #
## Master thesis for Nina Salvesen ##

This is the git repository for the master thesis for Nina Salvesen, in the field of applied physics and computer science at the Norwegian University of Science and Technology. The goal of the study is to find the normal and thus abnormal behavior of different Norwegian sheep breeds on outfield pastures, with the use of data driven analysis, statistics and machine learning. All code files are described below.

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

| Fosen         | Start date      | End date |
| :------------ |:---------------:| -----:   |
| 2018          | 03.06           | 29.06    |
| 2019          | 03.06           | 03.07 or 31.08    |
| 2020          | 03.06           | 05.09    |

| Tingvoll      | Start date (Koksvik/Torjul)| End date |
| :------------ |:---------------:| -----:              |
| 2012          | 09.06           | 07.09               |
| 2013          | 23.06 / 15.06   | 25.08               |
| 2014          | 05.06 / 25.06   | 10.09               |
| 2015          | 13.06 / 03.07   | 06.09               |
| 2016          | 17.06           | 22.07               |



