import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import UndefinedMetricWarning
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from multiprocessing import Pool, cpu_count
import os
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def evaluation(model_info, X_train, y_train, X_test, y_test, fold, train_period, test_period):
    name, model = model_info
    result = {}
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        result['name'] = name
        result['fold'] = fold
        result['train_period'] = train_period
        result['test_period'] = test_period
        result['accuracy'] = accuracy

        # AUC and ROC calculation
        try:
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        except Exception:
            auc = "N/A"
        result['auc'] = auc

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        result['report'] = report
    except Exception as e:
        result['error'] = str(e)

    return result

def main():
    outfile = 'output_report_1.txt'
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

    # Preparing features and target
    X = preprocessor.fit_transform(data_final)
    y = data_final['Target'].values

    # Setting up time series split
    tscv = TimeSeriesSplit(n_splits=5)

    # Models
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "CART Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        # "SVC": SVC(kernel='rbf', probability=True),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),   
        "CatBoost": CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, silent=True, random_state=42)
    }

    results = []

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_dates = data_final.iloc[train_index]['Date']
        test_dates = data_final.iloc[test_index]['Date']
        train_period = f"{train_dates.iloc[0]} - {train_dates.iloc[-1]}"
        test_period = f"{test_dates.iloc[0]} - {test_dates.iloc[-1]}"

        with Pool(processes=cpu_count()) as pool:
            fold_results = pool.starmap(evaluation, [(model_info, X_train, y_train, X_test, y_test, i+1, train_period, test_period) for model_info in models.items()])
            results.extend(fold_results)

    # Writing results to file
    with open(outfile, 'w') as file:
        for result in results:
            file.write(f"Fold {result['fold']} - Training Period: {result['train_period']}, Test Period: {result['test_period']}\n")
            file.write(f"{result['name']}\n")
            file.write(f"Accuracy: {result.get('accuracy', 'N/A')}\n")
            file.write(f"AUC: {result.get('auc', 'N/A')}\n")
            if 'report' in result:
                for label, metrics in result['report'].items():
                    if isinstance(metrics, dict):
                        for metric, value in metrics.items():
                            file.write(f"{label} {metric}: {value}\n")
            if 'error' in result:
                file.write(f"Error: {result['error']}\n")
            file.write("\n")

if __name__ == '__main__':
    main()
