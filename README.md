# AML Mini-Project: Predicting  Spotify Music Popularity Trends

## 1.Overview

In our mini-project, we aim to assist musicians in forecasting the future music popularity trends, specifically predicting whether their songs have the potential to reach the top 50 on the Spotify charts after a month.  Moreover, we seek to identify and evaluate the most influential factors shaping these trends: the impact of artists' profiles versus acoustic features.

[Data mining and crawling]

In our project, we utilize data mining and web crawling techniques, facilitated by the Spotify API, to collect release date for all songs and incorporate it into our dataset.

[Dataset]

- spotify_data_withrelease.csv
  - The original data set consists of 19 attributes with 651,936 rows of Spotify music from 2017.1.1 to 2023.5.31.
  - Collecting release date for all songs and incorporate it into our dataset by the Spotify API.

[Dataset Features]

- Title: Name of the songs.

- acoustic features: Danceability, Energy, Loudness, Speechiness, Acousticness, Instrumentalness, Valence 
- Points(Total): The total points which is calculate by ranking (Total points = 200 - ranking + 1).
- Popularity：whether a song's ranking enter the top 50
- id: The Spotify ID for the track.
- Artists: Artists of the tracks.
- Artist (Ind.)：the points that the artist own in the track.

[Feature Engineering]

- Superstar_artists: which means Whether the artist has a top 10 song
- Top50_Counts: which means The number of songs an artist made it into the top 50 before making the charts
- lagged_popularity: which means the popularity the songs own one month ago
- time_gap: which means the current date minus the release date


## 2.**EDA and Data Pre-processing**

### **Exporatory_Data_Analysis.ipynb**

You can directely  see this file to see how we process the data analysis,including the parts as follows.

Notice: This code is used to generate images for exploratory data analysis. If you need to run the code, please running 'run_data_preprocessing.py' before running this code (as generating Correlation between features requires this)

- input:
  - data_prepared_final.csv
  
- Exploratory Data Analysis
  - Numerical data description, Time series analysis, Data Visualization, Correlation between features

- Requirements: 
  - pandas 2.1.3
  - matplotlib 3.8.0
  - scipy 1.11.4
  - numpy 1.26.2
  - seaborn 0.12.2
  - scikit-learn 1.3.2
  
### **run_data_preprocessing.py**

Using command 'python3 run_data_preprocessing.py' to process the dataset

- input:
  - spotify_data_withrelease.csv
  
- output:
  - data_prepared_final.csv
  
- Dataset Description
  - Trend Analysis Scope & Prediction Analysis Scope

- Feature Engineering 
  - add features: 'Superstar', 'Top50_Counts','Popularity', 'Month_Label', 'Epoch', 'release_Epoch', 'Target', 'Popularity_lag_1'
  - delete features: '# of Artist', 'Artist (Ind.)', 'Points (Ind for each Artist/Nat)', 'Song URL', 'Artists', 'Nationality', '# of Nationality'

- Requirements: 
  - Pandas 2.1.3
  - Scikit-Learn 1.3.2

## 3.Model Training & Evaluation

### Folder:models

### **model_1/2/3.py**

Using command 'python model_1/2/3.py' to train and test models.

NOTICE: using command 'ulimit -n 8192' to expand the maximum opened files temporarily

Data mining techniques for popularity prediction in 3 different schemes. Each model_*.py file contains a feature engineering scheme, model training and evaluation procedure.

- input:
  - data_prepared_final.py

- output:
  - output_report_1/2/3.txt
  
- Requirements: 
  - pandas 2.1.3
  - scikit-learn 1.3.2
  - xgboost 1.7.3
  - catboost 1.2

##### Using different part of parameters

1. scheme_1: using all numerical features and categorical features except :'id', 'Title', 'release_date', 'Date', 'Month_Label' for feature engineering. In addition, 30 days of lagged feature are applied.
2. scheme_2: On the basis of scheme_1, removing all acoustic features, including: 'Danceability', 'Energy', 'Loudness_Normalized', 'Speechiness', 'Acousticness', 'Instrumentalness_Binary', 'Valence'
3. scheme_3: On the basis of scheme_1, removing all artists features, including: 'Superstar', 'Top50_Counts','Artist_Points'

##### Model Training

Popularity level prediction using classification models: Naive Bayse, Logistic Regression, Random Forest Classifier,  XGBoost,  CatBoost, SVM

##### Model Evaluation

Evaluating models with k-fold time series cross-validation. Various evaluation indices are calculated, they are Area Under Curve(AUC), Accuracy, Precision, recall, F1 Score, Support as well as their weighted and average value.

## 4.hyperparameter optimisation

### Folder:hyperparameter_optimisation

### **hyper_params_1/2/3.py**

Using command 'nohup python3 hyper_params_*.py > output_* &' to get and save the output log to a file named 'output_*', and '*' can be replaced as 1,2,3.  
Using grid search for hyperparameter optimisation.

- input:
  - data_prepared_final.csv

- output:
  - output_1, output_2, output_3
  
1. For each model, identify the hyperparameters which need to be optimised.
2. Evaluate the model's performance using k-fold time series cross-validation.
3. Implement grid search using scikit-learn library to find the best combinations.

- Requirements: 
  - pandas 2.1.3
  - scikit-learn 1.3.2
  - xgboost 1.7.3
  - catboost 1.2
