from sklearn import svm
import pandas
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
    return

if __name__ == "__main__":
    main()