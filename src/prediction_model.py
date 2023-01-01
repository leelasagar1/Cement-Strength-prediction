from data_loader import Data_Getter
from preprocessing import Preprocessor
import pandas as pd
import config
import os
import joblib


class Prediction:
    def find_correct_model_name(self, cluster):
        files = os.listdir(config.model_save_location)
        file_clust = [f.split("_")[-1][0] for f in files]
        print(file_clust)
        return files[file_clust.index(str(cluster))]

    def prediction_from_model(self):

        # get data from path
        data_getter = Data_Getter(
            config.pred_data_path, config.pred_schema_file)
        data = data_getter.get_data()

        # preprocess data
        preprocess = Preprocessor()
        null_present, cols_with_null = preprocess.is_null_present(data)

        if null_present:

            df = preprocess.impute_missing_values(data)

        df = preprocess.log_transformation(df)
        df_scaled = pd.DataFrame(
            preprocess.standard_scale_data(df), columns=df.columns)

        # predict clusters
        k_means_model = joblib.load(
            f"{config.model_save_location}/Kmeans_cluster.pkl")
        df_scaled["cluster"] = k_means_model.predict(df_scaled)

        # predict values for each cluster data
        result = pd.DataFrame()

        for cluster in df_scaled["cluster"].unique():

            cluster_data = df_scaled[df_scaled["cluster"] == cluster]

            model_name = self.find_correct_model_name(cluster)

            model = joblib.load(f"{config.model_save_location}/{model_name}")

            predictions = model.predict(cluster_data.drop("cluster", axis=1))
            cluster_data["predictions"] = predictions
            result = pd.concat([result, cluster_data["predictions"]])
            
        
        result = result.sort_index()
        print("------>Prediction Done")
        data["prediction"] = result
        data.to_csv(f"{config.output_data_path}/result.csv", index=False)
        return result
