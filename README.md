# Taxi-Demand-Forecasting-System

Winning solution of [The HerWILL Datathon 2022](https://www.kaggle.com/competitions/herwill-datathon-2022) hosted on by [HerWILL](https://herwill.org/).

## Task

The given training data is the record of **5 months (2017/01 - 2017/05)** of historical taxi trips data in a city. The data contains number of trips originating from **73 zones** in the city during each hour. The zones are indexed from 0 to 72.In addition to the taxi demand data, the weather and spatial information for the given dates and zones are also given.

**The task is to develop a model forecasting number of trips originating from each zone for the next hour.** 
For example, the model has to forecast trips for 7AM – 8AM at 7AM and forecast trips for 8AM – 9AM at 8AM. 

## Dataset Format

### Files
 * **2017-xx_1H_zone.csv** - Hourly trip data for a given month
     - PUZone: Trip's origin zone
     - Count: Number of trips originating from the zone during the given hour
     - PUTime: Date and hour of the pickup
 * **weather.csv** - Daily weather data. Description of the data is given in Table 4 in [weather_description.pdf](https://github.com/ramisa2108/Taxi-Demand-Forecasting-System/blob/main/data/weather_description.pdf)
 * **zone_neighbors.json** - Mapping between each zone and the list of zones adjacent to it.
 * **test.pkl.gz** - Compressed pickle file containing test dataset given as a list. Each entry in the list has
     - dt: The date and time for which the prediction has to be made
     - demand: A (24*30, 73) numpy array containing last 30 days' hourly demand data. 
               E.g. demand[-1, 3] contains last hour's demand of zone 3
     - weather: List of lists containing today's and last 30 days' weather data.
               E.g., weather[-1] is a list containing today's weather data with [DATE, AWND,...,WT08] as in weather.csv
     - neighbors: Dictionary containing the mapping between each zone and their list of neighbors in zone_neighbors.json

 * **test_answer.pkl.gz** - Compressed pickle file containing true demands for each entry of the test set.
                        
***Total number of entries in 'test.pkl.gz' and 'test_answer.pkl.gz' is equal***
   

