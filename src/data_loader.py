import json
import pandas as pd
import glob


class Data_Getter:
    def __init__(self, data_path, schema_file_path):
        self.data_path = data_path
        self.schema_file_path = schema_file_path

    # read schema_file
    def read_schema_file(self, path):

        f = open(path)
        schema = json.load(f)

        return schema

    # read_data from folder path
    def get_data(self):

        schema = self.read_schema_file(self.schema_file_path)
        columns = list(schema["ColName"].keys())
        data = [
            pd.read_csv(file,
                        names=columns, skiprows=1)
            for file in glob.glob(self.data_path+'/*.csv')
        ]
        data = pd.concat(data,ignore_index=True)
        return data
