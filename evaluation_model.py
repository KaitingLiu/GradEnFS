from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class SVM_Model():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    
    def train_and_test(self, selected_features):
        clf = svm.SVC()
        clf.fit(self.x_train[:, selected_features], np.argmax(self.y_train, axis=1))
        SVCacc = float(clf.score(self.x_test[:, selected_features], np.argmax(self.y_test, axis=1)))
        return SVCacc
    
class KNN_Model():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    
    def train_and_test(self, selected_features):
        clf = KNeighborsClassifier()
        clf.fit(self.x_train[:, selected_features], np.argmax(self.y_train, axis=1))
        KNNacc = float(clf.score(self.x_test[:, selected_features], np.argmax(self.y_test, axis=1)))
        return KNNacc
    
class ExtraTree_Model():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    
    def train_and_test(self, selected_features):
        clf = ExtraTreesClassifier()
        clf.fit(self.x_train[:, selected_features], self.y_train)
        ExtraTreeacc = float(clf.score(self.x_test[:, selected_features], self.y_test))
        return ExtraTreeacc