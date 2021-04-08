from sklearn import svm
import pandas as pd
import numpy as np
import random


def main():
    print("### Loading datasets")
    df_genre = pd.read_csv('data/title.basics.tsv',sep='\t')
    df_crew = pd.read_csv('data/title.crew.tsv',sep='\t')
    df_descriptions = pd.read_csv('data/movies_metadata.csv')

    # Get only the movies from genre dataset
    df_genre = df_genre[df_genre['titleType'] == 'movie']

    # Remove movies without a genre
    df_genre = df_genre[df_genre['genres'] != '\\N']
    
    # Merge genre data and crew data
    df_merged = pd.merge(left=df_genre, right=df_crew, left_on='tconst', right_on='tconst')
    df_merged = pd.merge(left = df_merged,
                right = df_descriptions[['imdb_id','overview']],
                left_on='tconst', right_on='imdb_id')
    print(len(df_merged))
    print(df_merged.head(5))
    df_merged.to_csv('df_merged.csv', encoding='utf-8', index=False)
    
    df_merged['genres'] = df_merged['genres'].apply(lambda x: x.split(','))
    for index, row in df_merged.iterrows():
       if len(row['genres']) != 1:
            df_merged.drop(index, inplace=True)
    print(df_merged)
    df_merged['genres'] = df_merged['genres'].apply(lambda x: ', '.join([str(i) for i in x]))
    df_merged.to_csv('df_single.csv', encoding='utf-8', index=False)     

if __name__ == "__main__":
    main()


