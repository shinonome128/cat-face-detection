"""
Load module
"""
import sys
import sklearn.svm
import pickle
from MacOSFile import pickle_dump, pickle_load
# import pdb; pdb.set_trace()


def main():

    # Read binary file of parameters and store in explanatory variable X, objective variable y
    # X, y = pickle.load(open(sys.argv[1], 'r+b'))
    # X, y = pickle.load(open("./get_feature.result", 'r+b'))
    X, y = pickle_load("./get_feature.result")

    # Create an instance of SVM
    classifier = sklearn.svm.LinearSVC(C = 0.0001)

    # Build a model by feeding an explanatory variable and an objective variable to an instance
    classifier.fit(X, y)

    # Save model as binary file
    # pickle.dump(classifier, open(sys.argv[2], 'wb'))
    pickle.dump(classifier, open("./make_model.result", 'wb'))


"""
This script will not be executed if called from another file
"""
if __name__ == "__main__":
    main()
