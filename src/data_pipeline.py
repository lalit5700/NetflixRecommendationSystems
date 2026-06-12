
import os
import numpy as np
import pandas as pd
import kagglehub

def download_and_parse_data(min_user_ratings=50, min_movie_ratings=500, sample_size=500000):
    print("🤖 Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("netflix-inc/netflix-prize-data")
    file_path = os.path.join(path, 'combined_data_1.txt')
    
    data_list = []
    print("📂 Parsing raw file combined_data_1.txt...")
    with open(file_path, 'r') as f:
        current_movie = None
        for line in f:
            line = line.strip()
            if line.endswith(':'):
                current_movie = int(line[:-1])
            else:
                parts = line.split(',')
                data_list.append([int(parts[0]), current_movie, int(parts[1])])
                
    df = pd.DataFrame(data_list, columns=['UserID', 'MovieID', 'Rating'], dtype=np.int32)
    print("⏳ Filtering long-tail noise and downsampling...")
    valid_users = df['UserID'].value_counts()[df['UserID'].value_counts() >= min_user_ratings].index
    valid_movies = df['MovieID'].value_counts()[df['MovieID'].value_counts() >= min_movie_ratings].index
    df_filtered = df[df['UserID'].isin(valid_users) & df['MovieID'].isin(valid_movies)]
    return df_filtered.sample(n=sample_size, random_state=42).reset_index(drop=True)
