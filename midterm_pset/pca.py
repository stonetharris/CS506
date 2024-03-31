import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

train_data = pd.read_csv('./voweltrain.csv', index_col=0)
test_data = pd.read_csv('./voweltest.csv', index_col=0)

X_train = train_data.filter(regex='x.*')
y_train = train_data['y']
X_test = test_data.filter(regex='x.*')
y_test = test_data['y']

class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        
        covariance_matrix = np.cov(X.T)

        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        eigenvectors = eigenvectors[:, np.argsort(-eigenvalues)]
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components)

class Component:
    def __init__(self, mixture_prop, mean, covariance):
        self.mixture_prop = mixture_prop
        self.mean = mean
        self.covariance = covariance

class GMMFromScratch:
    def __init__(self, n_components, n_iterations):
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.components = []

    def initialize(self, dataset):
        kmeans = KMeans(self.n_components, init='k-means++').fit(dataset)
        for j in range(self.n_components):
            p_cj = np.mean(kmeans.labels_ == j)
            mean_j = np.mean(dataset[kmeans.labels_ == j], axis=0)
            cov_j = np.cov(dataset[kmeans.labels_ == j].T)
            self.components.append(Component(p_cj, mean_j, cov_j))

    def expectation(self, dataset):
        self.probs = []
        for x in dataset:
            p_cj_x = np.array([c.mixture_prop * multivariate_normal.pdf(x, c.mean, c.covariance) for c in self.components])
            self.probs.append(p_cj_x / np.sum(p_cj_x))

    def maximization(self, dataset):
        for j in range(self.n_components):
            resp = np.array([p[j] for p in self.probs])
            total_resp = np.sum(resp)
            self.components[j].mixture_prop = total_resp / len(dataset)
            self.components[j].mean = np.sum(resp.reshape(-1, 1) * dataset, axis=0) / total_resp
            weighted_cov = sum([resp[i] * np.outer(dataset[i] - self.components[j].mean, dataset[i] - self.components[j].mean) for i in range(len(dataset))])
            self.components[j].covariance = weighted_cov / total_resp

    def fit(self, dataset):
        self.initialize(dataset)
        for _ in range(self.n_iterations):
            self.expectation(dataset)
            self.maximization(dataset)

    def predict_proba(self, dataset):
        self.expectation(dataset)
        return self.probs

    def predict(self, dataset):
        self.expectation(dataset)
        return np.argmax(self.probs, axis=1)


# pca = PCAFromScratch(n_components=2)
# pca.fit(X_train.values)
# X_train_pca = pca.transform(X_train.values)
# X_test_pca = pca.transform(X_test.values)
# classifier = RandomForestClassifier()
# classifier.fit(X_train_pca, y_train)
# y_pred = classifier.predict(X_test_pca)

# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy of PCA-transformed data with RandomForestClassifier: {accuracy:.4f}')
    
pca = PCAFromScratch(n_components=3)
pca.fit(X_train.values)

X_train_pca = pca.transform(X_train.values)
X_test_pca = pca.transform(X_test.values)

gmm = GMMFromScratch(n_components=11, n_iterations=100)
gmm.fit(X_train_pca)
gmm_labels = gmm.predict(X_test_pca) + 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

class_to_color_map = {
    1: "red",
    2: "blue",
    3: "green",
    4: "yellow",
    5: "orange",
    6: "purple",
    7: "pink",
    8: "brown",
    9: "gray",
    10: "olive",
    11: "cyan",
}

colors = [class_to_color_map[label] for label in gmm_labels]
ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], X_test_pca[:, 2], c=colors)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title('3D Scatter Plot of Vowel Data (Colored by GMM Labels)')
plt.show()