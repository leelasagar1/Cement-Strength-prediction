import yaml
import os

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

config_path = os.path.join("pipeline","params.yml")
config = read_params(config_path)
print(config)
