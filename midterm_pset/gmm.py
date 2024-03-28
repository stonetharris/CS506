import numpy as np
import pandas as pd
from numpy.random import uniform, normal
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

train_data = pd.read_csv('./voweltrain.csv', index_col=0)
test_data = pd.read_csv('./voweltest.csv', index_col=0)

X_train = train_data.filter(regex='x.*')
y_train = train_data['y']
X_test = test_data.filter(regex='x.*')
y_test = test_data['y']

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

gmm = GMMFromScratch(n_components=11, n_iterations=10)
gmm.fit(X_train.values)
predicted_labels = gmm.predict(X_test.values)
accuracy = accuracy_score(y_test, predicted_labels)
report = classification_report(y_test, predicted_labels, digits=4)
print(report)
print(f"Accuracy: {accuracy}")