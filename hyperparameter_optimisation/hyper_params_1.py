import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Data reading and pre-processing
data_final = pd.read_csv('../data_prepared_final.csv', sep=',')
data_final.sort_values(by='Date', inplace=True)

# Identifying categorical and numerical columns
categorical_cols = data_final.select_dtypes(include=['object', 'category']).columns.tolist()
unused_cols = ['id', 'Title', 'release_date', 'Date', 'Month_Label']
categorical_cols = [col for col in categorical_cols if col not in unused_cols]

numerical_cols = data_final.select_dtypes(include=['int64', 'float64']).columns.tolist()
unused_num_cols = ['Rank', 'Target', 'Popularity_Lag_2', 'Popularity_Lag_3']
numerical_cols = [col for col in numerical_cols if col not in unused_num_cols]

# Creating transformers for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# extracting features and target
X = preprocessor.fit_transform(data_final)
y = data_final['Target'].values

# set up time series split
tscv = TimeSeriesSplit(n_splits=5)

# set grids for searching
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_dt = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

param_grid_cat = {
    'iterations': [500, 1000, 1500],
    'depth': [4, 6, 10],
    'learning_rate': [0.01, 0.05, 0.1]
}

# initialise a dictionary for best hyperparameters' combination
best_params = {}

# grid search starts
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Random Forest
    grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, n_jobs=-1, verbose=2)
    grid_search_rf.fit(X_train, y_train)
    best_params['Random Forest'] = grid_search_rf.best_params_

    # Decision Tree
    grid_search_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=5, n_jobs=-1, verbose=2)
    grid_search_dt.fit(X_train, y_train)
    best_params['Decision Tree'] = grid_search_dt.best_params_

    # Logistic Regression
    grid_search_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, n_jobs=-1, verbose=2)
    grid_search_lr.fit(X_train, y_train)
    best_params['Logistic Regression'] = grid_search_lr.best_params_

    # XGBoost
    grid_search_xgb = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid_xgb, cv=5, n_jobs=-1, verbose=2)
    grid_search_xgb.fit(X_train, y_train)
    best_params['XGBoost'] = grid_search_xgb.best_params_

    # CatBoost
    grid_search_cat = GridSearchCV(CatBoostClassifier(silent=True), param_grid_cat, cv=5, n_jobs=-1, verbose=2)
    grid_search_cat.fit(X_train, y_train)
    best_params['CatBoost'] = grid_search_cat.best_params_

# print the best
print("Best Hyperparameters:")
for model, params in best_params.items():
    print(f"{model}: {params}")
