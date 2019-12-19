import sklearn
from sklearn import datasets
from sklearn import metrics
from sklearn import svm

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

classes = ['Malign', 'Bening']

classifier = svm.SVC(kernel="linear")  # linear kernel parameter makes the classifier fit better than the default rbf

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)