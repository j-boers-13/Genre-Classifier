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
    df_genre = pd.read_csv('data/title.basics.tsv',sep='\t')
    df_genre = df_genre[df_genre['titleType'] == 'movie']
    df_genre = df_genre[df_genre['genres'] != '\\N']
    print("reading crew")
    df_crew = pd.read_csv('data/title.crew.tsv',sep='\t')

    df_merged_full = pd.merge(left=df_genre, right=df_crew, left_on='tconst', right_on='tconst')
    df_merged = df_merged_full.sample(frac=0.5)
    # Writers and directors are a string with multiple actors/directors split by a comma
    # So first turn them into lists
    df_merged['genres'] = df_merged['genres'].apply(lambda x: x.split(','))
    df_merged = df_merged.explode('genres').fillna('')
    print(df_merged.head(5))

    # # List of features per movie
    # feature_samples = []
    # # List of labels per movie
    # labels = []
    one_hot_encoder = OneHotEncoder(categories='auto')
    features = one_hot_encoder.fit_transform(df_merged[['writers']])
    feature_labels = one_hot_encoder.categories_
    # print(feature_labels)
    # for _, row in df_merged.iterrows():
    #     for genre in row['genres'].split(','):
    #         labels.append(genre)
    #         features = row['writers'].split(',').extend(row['directors'].split(','))
    #         feature_samples.append(features)

    # train_features, test_features = split_train_test(feature_samples)
    # train_labels, train_features = split_train_test(labels)

    # mlb = MultiLabelBinarizer()

    # le = LabelEncoder()
    # le.fit(feature_samples)
    # print(le.classes_)


if __name__ == "__main__":
    main()
