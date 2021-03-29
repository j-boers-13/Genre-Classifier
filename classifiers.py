from sklearn import svm
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Train an svm classifier
# Hier kunnen we de SVM soort veranderen in:
# SVC, NuSVC, LinearSVC
def train_svm(feature_samples, labels):
    classifier = svm.SVC()
    classifier.fit(feature_samples, labels)
    return classifier

def split_train_test(feature_samples, split=0.8):
    """
    returns two feats
    
    splits a labelled dataset into two disjoint subsets train and test
    """
    train_feats = []
    test_feats = []

    random.Random(0).shuffle(feature_samples) # randomise dataset before splitting into train and test
    cutoff = int(len(feature_samples) * split)
    train_feats, test_feats = feature_samples[:cutoff], feature_samples[cutoff:]    

    print("\n##### Splitting datasets...")
    print("  Training set: %i" % len(train_feats))
    print("  Test set: %i" % len(test_feats))

    return train_feats, test_feats

def main():
    pd.set_option('display.max_columns', None)

    df_genre = pd.read_csv('data/title.basics.tsv',sep='\t')
    df_genre = df_genre[df_genre['titleType'] == 'movie']
    df_genre = df_genre[df_genre['genres'] != '\\N']
    df_crew = pd.read_csv('data/title.crew.tsv',sep='\t')

    df_merged_full = pd.merge(left=df_genre, right=df_crew, left_on='tconst', right_on='tconst')

    # Random sample half of the data
    df_merged = df_merged_full.sample(n=10000)

    # Split genres, writers and directors strings into array
    # These strings are formatted like genre1,genre2,genre3 etc.
    df_merged['genres'] = df_merged['genres'].apply(lambda x: x.split(','))
    # df_merged['writers'] = df_merged['writers'].apply(lambda x: x.split(','))
    # df_merged['directors'] = df_merged['directors'].apply(lambda x: x.split(','))

    # # Create split directors df
    # df_directors = pd.DataFrame([pd.Series(x) for x in df_merged.directors])
    # df_directors.columns = ['director_{}'.format(x+1) for x in df_directors.columns]

    # # Create split writers df
    # df_writers = pd.DataFrame([pd.Series(x) for x in df_merged.writers])
    # df_writers.columns = ['writer_{}'.format(x+1) for x in df_writers.columns]

    # Create a duplicate row per genre (split genres)
    df_merged = df_merged.explode('genres').fillna('')

    # # Reset indices for joining
    # df_merged = df_merged.reset_index()
    # df_directors = df_directors.reset_index()
    # df_writers = df_writers.reset_index()

    # # Join the dataframes
    # df_merged = pd.concat([df_merged, df_directors], axis=1)
    # df_merged = pd.concat([df_merged, df_writers], axis=1)

    # Replace \N string by np.nan (moet dit None zijn?)
    df_merged.replace("\\N", np.nan)
    
    column_names = df_merged.filter(regex='writer|director').columns.tolist()
    column_names = [name for name in column_names if name not in ['writers', 'directors']]

    # Convert columns dtype from objects to strings
    # For memory optimization
    df_merged[column_names] = df_merged[column_names].astype(str)

    # Get one hot encoded sparse-matrix
    # Deze kan beter dealen met NA values, ik heb nu \N ook NA gemaakt,
    # dus kan die alles handlen
    feature_matrix = pd.get_dummies(df_merged[['writers','directors']], dummy_na=True)

    # Get list of genres per observation
    labels = df_merged['genres'].tolist()

    # Train SVM model
    classifier = train_svm(feature_matrix, labels)


if __name__ == "__main__":
    main()
