
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score, classification_report, adjusted_rand_score, homogeneity_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from ucimlrepo import fetch_ucirepo
label_encoder = LabelEncoder()


def model_test(X, y, noClusters):

    kmeans=KMeans(n_clusters=noClusters, random_state=42)
    clusters=kmeans.fit_predict(X)
    X_supervised = X.copy()
    y_supervised = clusters
    print(len(X_supervised))

    X_train, X_test, y_train, y_test = train_test_split(X_supervised, y_supervised, test_size=0.25, random_state=42)
    tree=DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    ari = adjusted_rand_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Adjusted Rand Index: {ari}")

print("Dry bean dataset: ")
dry_bean = fetch_ucirepo(id=602)
model_test(dry_bean.data.features, dry_bean.data.targets, 7)
print("\n\n")
data = load_iris()
X = data.data
y = data.target
print("Iris dataset: ")
model_test(X, y, 3)
print("\n\n")

data = load_wine()
X = data.data
y = data.target
print("Wine dataset: ")
model_test(X, y, 3)
print("\n\n")
data = load_digits()
X = data.data
y = data.target
print("Digits dataset: ")
model_test(X, y, 10)


