from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1)

clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(X_train, y_train)

print(clf.feature_importances_)

y_pred = clf.predict(X_test)

roc = roc_auc_score(y_test, y_pred)

print(f'ROC_AUC: {roc}')