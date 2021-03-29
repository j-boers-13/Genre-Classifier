from sklearn import svm
import pandas as pd
import random
from sklearn.feature_extraction import FeatureHasher

# Train an svm classifier.
# SVC, NuSVC, LinearSVC
def train_svm(feature_samples, labels):
    classifier = svm.SVC()
    classifier.fit(feature_samples, labels)
    return classifier

# Create feature array for
# movie based on writers and directors.
# Uses yield for lazy evaluation.
# This causes the FeatureHasher to
# only call a feature when it needs it.
def movie_features(writers, directors):
  for writer in writers.split(","):
      yield "writer={}".format(writer.lower())
  for director in directors.split(","):
      yield "director={}".format(director.lower())

def main():
    # Load datasets
    print("### Loading datasets")
    df_genre = pd.read_csv('data/title.basics.tsv',sep='\t')
    df_crew = pd.read_csv('data/title.crew.tsv',sep='\t')

    # Get only the movies from genre dataset
    df_genre = df_genre[df_genre['titleType'] == 'movie']

    # Remove movies without a genre
    df_genre = df_genre[df_genre['genres'] != '\\N']
    
    # Merge genre data and crew data
    df_merged = pd.merge(left=df_genre, right=df_crew, left_on='tconst', right_on='tconst')

    # Take a random sample of 50% to cut the data in half
    df_merged = df_merged.sample(n=5000)

    # Make genres a list instead of string genre1,genre2,genre3
    df_merged['genres'] = df_merged['genres'].apply(lambda x: x.split(','))

    # Split genres and give them each their own row with duplicate values
    df_merged = df_merged.explode('genres')

    print("Observations: {}".format(len(df_merged)))

    # Get array of features for each observation
    print("### Creating features and labels")
    raw_X = (movie_features(row.directors, row.writers) for index, row in df_merged.iterrows())

    # Get genre for each observation
    y = df_merged['genres'].tolist()

    # Create hash encoding for features, to lower memory use
    print("### Creating hash encoding for features")
    hasher = FeatureHasher(input_type='string')
    X = hasher.fit_transform(raw_X, y)

    # Train classifier
    print("### Training SVM classifier")
    classifier = train_svm(X, y)

if __name__ == "__main__":
    main()