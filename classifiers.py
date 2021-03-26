from sklearn import svm
import pandas
import pandas as pd

#from constants import TITLE_BASICS, TITLE_CREW

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
    df = pd.read_csv('title.basics.csv',sep='\t')
    df2 = df[df.titleType == 'movie']
    print(df2)
    #data1 = pd.read_csv('title.crew.csv',sep='\t')
    #print(data1)    
    return

if __name__ == "__main__":
    main()
