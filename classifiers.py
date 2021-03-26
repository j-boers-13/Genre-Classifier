from sklearn import svm
import pandas as pd
from constants import TITLE_BASICS, TITLE_CREW

# Train an svm classifier
# Hier kunnen we de SVM soort veranderen in:
# SVC, NuSVC, LinearSVC
def train_svm(feature_samples, labels):
    classifier = svm.SVC()
    classifier.fit(feature_samples, labels)
    return classifier

def get_features_labels(data):
    return

def main():
    df_genre = pd.read_csv(TITLE_BASICS,sep='\t')
    df_genre2 = df_genre[df_genre.titleType == 'movie']
    df_genre2 = df_genre2[df_genre.genres != '\\N']
    #print(df_genre2)
    df_crew = pd.read_csv('title.crew.tsv',sep='\t')
    #print(df_crew)
    df_merged = pd.merge(left=df_genre2, right=df_crew, left_on='tconst', right_on='tconst')
    print(df_merged)
    return

if __name__ == "__main__":
    main()
