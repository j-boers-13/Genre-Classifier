from sklearn import svm
import pandas as pd
import random
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from random import shuffle
import xgboost
import pickle
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sn

def gbm_gridsearch(X, y, nfolds):
    param_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [1, 1.5, 2],
        'max_depth': [3, 4, 5]
    }

    grid_search = GridSearchCV(xgboost.XGBClassifier(verbosity = 0), param_grid, cv=nfolds, n_jobs=2, refit = True, verbose = 3)    

    grid_search.fit(X, y)

    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    optimised_gbm = grid_search.best_estimator_
    print(grid_search.best_params_)

    return optimised_gbm

def svm_gridsearch(X, y, nfolds):
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    grid_search = GridSearchCV(svm.SVC(), param_grid, n_jobs=-1, cv=nfolds, refit = True, verbose = 3)

    grid_search.fit(X, y)

    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    optimised_svm = grid_search.best_estimator_
    print(grid_search.best_params_)

    return optimised_svm

def train_gbm(feature_samples, labels):
    classifier = xgboost.XGBClassifier()
    classifier.fit(feature_samples, labels)
    
    return classifier

# Train an svm classifier.
# SVC, NuSVC, LinearSVC
def train_svm(feature_samples, labels):
    classifier = svm.SVC()
    classifier.fit(feature_samples, labels)

    return classifier

# Deprecated
def shuffle_sparse_matrix(sparse_feats):
    indices = np.arange(sparse_feats.shape[0]) #gets the number of rows 
    shuffle(indices)
    shuffled_matrix = sparse_feats[list(indices)]

    return shuffled_matrix

# Split dataframe in train dev and test sets
def split_train_dev_test(df):
    """
    returns two feats
    
    splits a labelled dataset into two disjoint subsets train and test
    """

    df = df.sample(frac=1) # randomise dataset before splitting into train and test

    train_cutoff = int(len(df) * 0.8)
    train, non_train = df[:train_cutoff], df[train_cutoff:]

    dev_test_cutoff = int(len(non_train) * 0.5)
    dev, test = non_train[:dev_test_cutoff], non_train[dev_test_cutoff:]

    print("\n##### Splitting dataset...")
    print("  Training set: %i" % len(train))
    print("  Dev set: %i" % len(dev))
    print("  Test set: %i" % len(test))

    return train, dev, test

# Create feature array for
# movie based on writers and directors.
# Uses yield for lazy evaluation.
# This causes the encoder to
# only call a feature when it needs it.
def movie_features(writers, directors):
  for writer in writers.split(","):
      yield "writer={}".format(writer.lower())
  for director in directors.split(","):
      yield "director={}".format(director.lower())

def main():
    # Load datasets
    print("\n### Loading datasets")
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

    print("\n### Preview of the dataset")
    print(len(df_merged))
    print(df_merged.head(5))

    
    # Make genres a list instead of string genre1,genre2,genre3
    df_merged['genres_split'] = df_merged['genres'].apply(lambda x: x.split(','))

    # Take only rows with a single genre
    df_merged = df_merged[(df_merged['genres_split'].map(len) == 1)]

    print("Observations: {}".format(len(df_merged)))

    train_df, dev_df, test_df = split_train_dev_test(df_merged)

    print("\n### Creating features and labels")
    hasher = FeatureHasher(input_type='string')
    encoder = LabelEncoder()
    encoder.fit(df_merged['genres'].tolist())
    raw_X = (movie_features(row.directors, row.writers) for index, row in df_merged.iterrows())
    hasher.fit(raw_X)

    raw_X_train = (movie_features(row.directors, row.writers) for index, row in train_df.iterrows())
    y_train = encoder.transform(train_df['genres'].tolist())
    X_train = hasher.transform(raw_X_train)

    raw_X_dev = (movie_features(row.directors, row.writers) for index, row in dev_df.iterrows())
    y_dev = encoder.transform(dev_df['genres'].tolist())
    X_dev = hasher.transform(raw_X_dev)

    raw_X_test = (movie_features(row.directors, row.writers) for index, row in test_df.iterrows())
    y_test = encoder.transform(test_df['genres'].tolist())
    X_test = hasher.transform(raw_X_test)
    
    print("\n### Training SVM classifier on best params")
    svm_clf = svm_gridsearch(X_train, y_train, 2)
    pickle.dump(svm_clf, open("svm.pickle.dat", "wb")) # Way to save and load models
    svm_y_pred = svm_clf.predict(X_test)

    print(classification_report(y_test, svm_y_pred))
    svm_cm = confusion_matrix(y_test, svm_y_pred)
    unique_labels = np.unique(test_df['genres'].tolist() + list(encoder.inverse_transform(svm_y_pred)))
    df_svm_cm = pd.DataFrame(svm_cm, index = unique_labels,
                  columns = unique_labels)
    sn.heatmap(df_svm_cm, annot=True, fmt='g')
    plt.show()

    print("\n### Training GBM classifier")
    gbm_clf = gbm_gridsearch(X_train, y_train, 2)
    pickle.dump(gbm_clf, open("gbm.pickle.dat", "wb")) # Way to save and load models
    gbm_y_pred = gbm_clf.predict(X_test)

    print(classification_report(y_test, gbm_y_pred))
    gbm_cm = confusion_matrix(y_test, gbm_y_pred)
    unique_labels = np.unique(test_df['genres'].tolist() + list(encoder.inverse_transform(gbm_y_pred)))
    df_gbm_cm = pd.DataFrame(gbm_cm, index = unique_labels,
                  columns = unique_labels)
    sn.heatmap(df_gbm_cm, annot=True, fmt='g')
    plt.show()

if __name__ == "__main__":
    main()