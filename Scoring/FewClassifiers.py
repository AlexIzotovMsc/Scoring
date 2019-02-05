# Logit
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(n_jobs=-1)
logr.fit(X_train, y_train)
y_pred_test = logr.predict(X_test)

print('Accuracy: % .2f' % accuracy_score(y_test, y_pred_test))

CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
print(CM)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
class_tree = DecisionTreeClassifier(min_samples_split=30, min_samples_leaf=10, random_state=0)
class_tree.fit(X_train, y_train)
y_pred_test = class_tree.predict(X_test)

print('Accuracy: % .2f' % accuracy_score(y_test, y_pred_test))

CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
print(CM)


# MultiLayer Perceptrone
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(max_iter = 1000, activation = 'relu')
clf.fit(X_train, y_train)
y_pred_test = clf.predict(X_test)

print('Accuracy: % .2f' % accuracy_score(y_test, y_pred_test))

CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
print(CM)
