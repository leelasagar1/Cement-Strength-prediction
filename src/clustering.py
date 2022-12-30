from sklearn.cluster import KMeans
from kneed import KneeLocator
import joblib
import os
import config


class KMean_Clustering:
    def __init__(self):
        self.kmeans_kwargs = {
            "init": "random",
            "n_init": 10,
            "max_iter": 300,
            "random_state": 42,
        }
        self.filepath = config.model_save_location
        self.filename = "Kmeans_cluster.pkl"

    def find_number_of_clusters(self, data):
        wcss = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, **self.kmeans_kwargs)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)

        kn = KneeLocator(range(1, 11), wcss, curve="convex",
                         direction="decreasing")
        return kn.elbow

    def create_clusters(self, data, number_of_clusters):

        kmeans = KMeans(n_clusters=number_of_clusters, **self.kmeans_kwargs)
        pred = kmeans.fit_predict(data)
        data["cluster"] = pred

        joblib.dump(kmeans, os.path.join(self.filepath, self.filename))

        return data
