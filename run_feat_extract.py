import importlib
import os

import dotenv
import hydra
from omegaconf import DictConfig

dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/feat_extract/", config_name="feat_extract.yaml")
def main(config: DictConfig):
    input_model = config.params.input_model
    print("input_model: ", input_model)
    # print("Config.params:")
    # print(config.params)
    module = importlib.import_module(config._target_)
    module.create_csv(**dict(config.params, input_model=input_model))


if __name__ == "__main__":
    main()
