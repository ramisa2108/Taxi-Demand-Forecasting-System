# Taxi-Demand-Forecasting-System

A machine learning based forecasting system for predicting taxi demand in a city.

Winning solution of [The HerWILL Datathon 2022](https://www.kaggle.com/competitions/herwill-datathon-2022) hosted on by [HerWILL](https://herwill.org/).

## Task

The given training data is the record of **5 months (2017/01 - 2017/05)** of historical taxi trips data in a city. The data contains number of trips originating from **73 zones** in the city during each hour. The zones are indexed from 0 to 72.In addition to the taxi demand data, the weather and spatial information for the given dates and zones are also given.

**The task is to develop a model forecasting number of trips originating from each zone for the next hour.** 
For example, the model has to forecast trips for 7AM – 8AM at 7AM and forecast trips for 8AM – 9AM at 8AM. 

The evaluation metric is MAE (Mean Absolute Error) in this task. For the test dataset, the overall MAE and zone wise MAE for all 73 zones have to be predicted.

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
   
## Solution Approach

### Exploratory Data Analysis

Some insights found through EDA performed in [EDA.ipynb](https://github.com/ramisa2108/Taxi-Demand-Forecasting-System/blob/main/EDA.ipynb):

* The ACF and PACF plots for trip count showed higher correlation with recent previous hours and has almost a periodic tendency of 24 hours.
* Two peak traffic periods *(06:00:00-10:00:00 & 16:00:00-20:00:00)* were noticed in week days and one *(16:00:00-20:00:00)* in weekends.
* Neighbour zones are not highly correlated with each other. In fact in many cases they showed low correlation.
* Correlation analysis of weather parameters showed that most of them had very little correlation with traffic. The most correlated features we could find were **snow, snow depth, fog, heavy fog**.
* Some zones had higher average traffic than others.

### Feature Selection

Based on our findings we selected the following features for our models:

1. **Traffic demand in the past**: From our findings in the ACF and PACF plots, we used traffic demands in the <ins>*last 24 hours*</ins> for indicating today’s demand and traffic demands of the <ins>*previous 2 hour window for the last 30 days*</ins> for indicating that hour’s overall demand.
2. **Weekday/Weekend**: By plotting traffic against day-of-the-week, we observed higher traffic on weekdays. So we added a feature indicating whether the given day is a weekday.
3. **Peak hour**: Based on the peak hours mentioned above we created a new feature by combining *"Weekday"* and *"hour"* features.
4. **High Traffic Zone**: Based on our analysis, we added a feature indicating whether the given zone usually has heavy traffic.
5. We tried using the weather parameters most correlated to traffic (snow, fog etc.). But none of these impacted the performance of our models much. So we decided to not include any of them as features.
6. The given information on neighbours data did not provide any useful insight. So we did not use neighbour data as features.

### Models and Parameters

We used the first four months of the dataset as training data, rest was used for validation. We also cross validated our model using special cross validation techniques for time series models. Initially, we trained some time-series forecasting models such as: ARIMA, Prophet etc. but the results obtained were unsatisfactory. So we trained the data on regression models such as Xgboost, LightGBM, Multilayer Perceptron(MLP) and Random Forest. Among these models the performance of MLP and Random Forest were great with both yielding satisfactory MAE. Even though LightGBM wasn’t the best in terms of accuracy, its speed helped us with testing. Along with trying various models, we also tried parameter tuning with LightGBM, Xgboost, MLP and Random Forest by increasing the number of estimators and max iterations. Increasing these further tended to overfit the model. Finally we used an ensemble of **Xgboost, LightGBM, Multilayer Perceptron(MLP) and Random Forest with Voting Regressor**.

### Results and Discussion

| Model | Tuned Parameters | Mean Absolute Error | Findings |
| ----- | ---------------- | ------------------- | ---------|
| Voting Regressor Ensemble | Models: LGB, XGB, MLP and Random Forest |    **14.63** | Best Performance |
| LightGBM | Larger max bin, smaller learning rate | 15.37 | Very fast, useful for testing, MAE satisfactory |
| XgBoost | Increased number of estimators | 15.17 | Performed well |
| Multilayer Perceptron | Increased max iteration and hidden layer dimension | 15.05 | Performed very well, but very slow |
| Random Forest | Increased number of estimators | 15.19 | Performed very well, but very slow and high number of estimators consumes too much memory |
| Prophet | Default | >Baseline | Poor performance |
| Decision Tree Regressor| Increased max depth | >Baseline | Poor performance |
| Support Vector Regression | Standard Scaling | >Baseline | Poor performance |
| Ridge Regression | Default | >Baseline | Poor performance |
Gradient Boosting Regressor | Default | >Baseline | Poor performance |


Our best performing model yielded an MAE score of **14.63**. The performance of other models are described in the above table.

