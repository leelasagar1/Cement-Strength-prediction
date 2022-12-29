from data_loader import Data_Getter
from preprocessing import Preprocessor
from clustering import KMean_Clustering
from best_model_finder import Model_Finder
from sklearn.model_selection import train_test_split
import config


class Train_model:
    def train(self):

        # get data
        data_getter = Data_Getter(config.train_data_path, config.train_schema_file)
        data = data_getter.get_data()

        # preprocess data
        preprocessor = Preprocessor()
        # data = preprocessor.remove_nan(data)
        data = preprocessor.impute_missing_values(data)
        X, y = preprocessor.seperate_target_feature(
            data, target_column="Concrete_compressive_strength"
        )
        X = preprocessor.log_transformation(X)
        print("------->Preprocessing Done")
        # find clusters in data
        kmeans_cluster = KMean_Clustering()
        num_of_clusters = kmeans_cluster.find_number_of_clusters(data)
        X = kmeans_cluster.create_clusters(X, number_of_clusters=num_of_clusters)
        X["label"] = y
        print("------->Clustering Done")
        # find best models for each cluster
        print("------->Model training started")
        for cluster in X["cluster"].unique():

            cluster_features = X.drop(["cluster", "label"], axis=1)
            cluster_label = X["label"]

            x_train, x_test, y_train, y_test = train_test_split(
                cluster_features, cluster_label, test_size=0.3, random_state=42
            )

            x_train_scaled = preprocessor.standard_scale_data(x_train)
            x_test_scaled = preprocessor.standard_scale_data(x_test)

            model_finder = Model_Finder()

            best_model_name, best_model = model_finder.find_best_model(
                (x_train_scaled, x_test_scaled, y_train, y_test)
            )
            model_finder.save_model(best_model_name, best_model, cluster)
        print("------->Model training Done")
