import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
import os

def preprocess_anime_data(df):
    # Handle potential missing values in 'genre' 
    df['genre'] = df['genre'].fillna('')

    # Split the genres and create a list of all unique genres 
    genres_list = df['genre'].str.split(', ').explode().unique()

    # Create dummy variables for each genre 
    for genre in genres_list:
        # Ensure genre is a string before creating column name
        if isinstance(genre, str):
            df[f'genre_{genre}'] = df['genre'].str.contains(genre, na=False).astype(int)

    # Drop the original 'genre' column 
    df.drop(['genre'], axis=1, inplace=True)

    # One-Hot Encode 'type' column 
    df = pd.get_dummies(df, columns=['type'], prefix='type')

    # Label Encode 'rank' column 
    label_encoder = LabelEncoder()
    df['rank_encoded'] = label_encoder.fit_transform(df['rank'])

    # Select and scale numerical features using RobustScaler 
    features_to_scale = df[['rating', 'members', 'episodes']]
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features_to_scale)
    scaled_df = pd.DataFrame(scaled_features, columns=['rating_scaled', 'members_scaled', 'episodes_scaled'])

    # Drop original numerical columns and concatenate scaled features
    df = df.drop(['rating', 'members', 'episodes'], axis=1)
    df = pd.concat([df, scaled_df], axis=1)
    return df

if __name__ == '__main__':
    try:
        raw_df = pd.read_csv('dataset_anime.csv')
        processed_df = preprocess_anime_data(raw_df.copy())
        output_dir = 'preprocessing'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/dataset_anime_clean.csv'
        processed_df.to_csv(output_path, index=False)
    except FileNotFoundError:
        print("Error: dataset_anime.csv not found. Please make sure the dataset is in the same directory.")