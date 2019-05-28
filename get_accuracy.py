"""
Load module
"""
import sys
import pickle
# import sklearn.svm


def main():

        # Read from the feature file of trained SVM model and test data and read from binary file format
	# classifier = pickle.load(open(sys.argv[1], 'r+b'))
	classifier = pickle.load(open("./make_model.result", 'r+b'))
	# X, y = pickle.load(open(sys.argv[2], 'r+b'))
	X, y = pickle.load(open("./get_feature.result_test", 'r+b'))

        # pass features to predict to predict labels
	y_predict = classifier.predict(X)

        # Find the correct answer rate by comparing the predicted label with the correct answer
	correct = 0
	for i in range(len(y)):
            if y[i] == y_predict[i]: correct += 1
	print('Accuracy: %f' % (float(correct) / len(y)))


"""
This script will not be executed if called from another file
"""
if __name__ == "__main__":
    main()
