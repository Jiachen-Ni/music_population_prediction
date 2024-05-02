import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas.tseries.offsets import MonthEnd
import datetime

def main():
    # Load the dataset with the 'release date'
    file_path = 'spotify_data_withrelease.csv'
    spotify_data = df = pd.read_csv(file_path, sep=',')

    # Add a feature representing the average points of each artist
    # Calculate the mean points for each artist
    artist_points = spotify_data.groupby('Artists')['Points (Ind for each Artist/Nat)'].mean().reset_index()
    # Rename the columns appropriately
    artist_points.rename(columns={'Points (Ind for each Artist/Nat)': 'Artist_Avg_Points'}, inplace=True)
    # Merge the average points with the original dataframe
    spotify_data = spotify_data.merge(artist_points, on='Artists', how='left')

    # add a feature 'superstar'
    spotify_data_super = create_superstar_feature(spotify_data, cutoff_date='2023-05-31')

    # add a feature 'top50'
    spotify_data_super = create_top50(spotify_data_super, cutoff_date='2023-05-31')

    # Delete duplicate rows based on columns' id ',' Date ', and' Artist (Ind.) ',
    # keeping the first row of each group of duplicate rows
    data_del = spotify_data_super.drop_duplicates(subset=['id', 'Date', 'Artist (Ind.)'])

    # Weighted average of author scores
    score_sum = data_del.groupby('id')['Artist_Avg_Points'].sum()
    data_del = data_del.merge(score_sum.rename('Artist_Points'), on='id')
    # Delete duplicate rows based on columns'Title', 'id', 'Date',
    data_del = data_del.drop_duplicates(subset=['Title', 'id', 'Date'])

    # Delete useless columns
    columns_to_remove = ['# of Artist', 'Artist (Ind.)', 'Points (Ind for each Artist/Nat)', 'Song URL',
                         'Artists', 'Nationality', '# of Nationality', 'Artist_Avg_Points']
    data_del.drop(columns=columns_to_remove, inplace=True)

    # Create month timestamp
    # Function to assign a label for each month
    data_del['Date'] = pd.to_datetime(data_del['Date'])
    data_del['Month_Label'] = data_del['Date'].dt.to_period('M')
    data_del['Month_End'] = pd.to_datetime(data_del['Date']) + MonthEnd(0)

    # Create a polupar column, where the top 50 is set to 1 at least once in the month, otherwise it is set to 0
    data_pop = popular(data_del)

    data = data_pop.copy()
    # Ensure the Date column is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Convert the Month_Label into a period type for easier calculations
    data['Month_Label'] = data['Month_Label'].apply(lambda x: pd.Period(x, freq='M'))

    # Create a period column for the date to match with Month_Label
    data['Date_Period'] = data['Date'].dt.to_period('M')

    # add lagged feature and target feature
    data_with_lags = create_lagged_popularity(data, [1, 2, 3, -1])
    data_with_lags = data_with_lags.rename(columns={'Popularity_Lag_-1': 'Target'})

    # Loudness normalization, instrumentation binarization
    data_final = lound_instru(data_with_lags)

    # Apply one-hot encoding to the 'Continent' feature
    # Use get_dummies to perform one-hot encoding
    continent_dummies = pd.get_dummies(data_final['Continent'], prefix='Continent')

    # Join the one-hot encoded columns back to the original DataFrame
    data_final = pd.concat([data_final, continent_dummies], axis=1)

    # Convert data to epoch
    data_final['Date'] = pd.to_datetime(data_final['Date'])
    data_final['Epoch'] = data_final['Date'].astype('int64') // 10 ** 9
    data_final['release_date'] = pd.to_datetime(data_final['release_date'])
    data_final['release_Epoch'] = data_final['release_date'].astype('int64') // 10 ** 9
    data_final = data_final.drop(columns=['release_date', 'Month_End', 'Date_Period'])
    data_final.to_csv(f'data_prepared_final.csv', index=False)

    # Function to create lagged features


def create_lagged_popularity(df, lag_months):
    # Creating a copy of the dataframe to avoid modifying the original data
    lagged_df = df.copy()
    for lag in lag_months:
        # Calculate the period for lag
        lagged_df[f'Month_Label_Lag{lag}'] = lagged_df['Month_Label'].apply(lambda x: x - lag)

        # Map the lagged period and id to the popularity
        popularity_map = lagged_df.set_index(['id', 'Month_Label'])['Popularity'].to_dict()
        lagged_df[f'Popularity_Lag_{lag}'] = lagged_df.apply(
            lambda row: popularity_map.get((row['id'], row['Month_Label_Lag' + str(lag)]), 0), axis=1)
    df.rename(columns={'Popularity_Lag_-1': 'Popularity_target'}, inplace=True)
    # Drop the intermediate columns
    lagged_df.drop(columns=[f'Month_Label_Lag{lag}' for lag in lag_months], inplace=True)
    return lagged_df


# superstar
def create_superstar_feature(df, cutoff_date):
    # Convert the date column to datetime type
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Remove leading and trailing whitespaces from all column names
    df.columns = df.columns.str.strip()

    # Find the top 50 songs and filter out songs before the cutoff date
    top_songs = df[(df['Rank'] <= 5) & (df['Date'] < pd.to_datetime(cutoff_date))]

    # Get the number of songs each artist had in the top charts before the cutoff date
    artist_top_chart_counts = top_songs.groupby('Artist (Ind.)').size().reset_index(name='Counts')

    # Identify artists who had at least one song in the top 50 before the cutoff date
    superstar_artists = set(artist_top_chart_counts['Artist (Ind.)'].unique())

    # Create a Superstar feature
    # Check if each song's artist is in the list of superstar artists
    df['Superstar'] = df['Artist (Ind.)'].apply(
        lambda x: any(artist in superstar_artists for artist in x.split(', '))
    )

    return df


# top 50
def create_top50(df, cutoff_date):
    # Convert the date column to datetime type
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Remove leading and trailing whitespaces from all column names
    df.columns = df.columns.str.strip()

    # Find songs ranked in the top 50 and filter out songs before the cutoff date
    top_songs = df[(df['Rank'] <= 50) & (df['Date'] < pd.to_datetime(cutoff_date))]

    # Get the number of times each artist appeared in the charts before the cutoff date
    artist_chart_counts = top_songs.groupby('Artist (Ind.)').size().reset_index(name='Top50_Counts')

    # Merge the count of chart appearances back into the original dataset
    df = df.merge(artist_chart_counts, on='Artist (Ind.)', how='left')

    # Fill in 0 for artists who did not appear in the charts
    df['Top50_Counts'].fillna(0, inplace=True)

    return df


def popular(spotify_data):
    # Define songs that appeared at least once in the top 50 in a month as popular, marked as 1
    # Group by Title and determine if each song has ever scored points indicating a top 50 appearance
    popularity_by_quarter = spotify_data.groupby(['Title'])['Points (Total)'].max().reset_index()

    # Label as 1 if the song scored enough points to indicate a top 50 appearance, else 0
    popularity_by_quarter['Popularity'] = popularity_by_quarter['Points (Total)'].apply(lambda x: 1 if x >= 151 else 0)

    # Merge this information back to the original dataframe
    spotify_data_new = spotify_data.merge(popularity_by_quarter[['Title', 'Popularity']], on=['Title'], how='left')

    # Drop the old Popularity column and keep the new one
    spotify_data_new = spotify_data_new.rename(columns={'Popularity_y': 'Popularity'})
    return spotify_data_new


# add lagged feature
def add_lag(data):
    # Convert Month_Label to the last day of the month for easier comparison

    # Creating lagged features for 1, 2, and 3 months
    for lag in range(1, 4):
        # Shifting the 'Popularity' column by 1, 2, and 3 months
        data[f'Popularity_Lag_{lag}'] = data.groupby('id')['Popularity'].shift(lag)

        # Comparing the shifted 'Popularity' with the month's last date
        # If the 'Date' in shifted rows is within the lagged month, keep the popularity, else set to 0
        data[f'Popularity_Lag_{lag}'] = data.apply(
            lambda row: row[f'Popularity_Lag_{lag}']
            if (row['Date'] >= row['Month_End'] - datetime.timedelta(days=30 * lag))
               and (row['Date'] <= row['Month_End']) - datetime.timedelta(days=30 * (lag - 1))
            else 0, axis=1)
    return data


# Loudness normalization, instrumentation binarization
def lound_instru(spotify_data):
    scaler = MinMaxScaler()
    loudness = spotify_data['Loudness'].values.reshape(-1, 1)
    spotify_data['Loudness_Normalized'] = scaler.fit_transform(loudness)
    # instrumentalness binarization
    # Binarize the 'Instrumentalness' feature
    # Define a threshold to binarize the 'Instrumentalness' feature
    threshold = 0
    spotify_data['Instrumentalness_Binary'] = (spotify_data['Instrumentalness'] > threshold).astype(int)
    spotify_data = spotify_data.drop(columns=['Instrumentalness', 'Loudness'])
    return spotify_data


if __name__ == "__main__":
    main()