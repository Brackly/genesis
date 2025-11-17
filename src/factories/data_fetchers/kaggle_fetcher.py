import configs.config as config
from pathlib import Path
import kagglehub
from src.factories.data_fetchers import Fetcher

class KaggleFetcher(Fetcher):
    def __init__(self,data_config:type(config.DataConfig)):
        super().__init__()
        self.local_save_path = None
        self.config : config.DataConfig = data_config

    def fetch(self):
        save_path = kagglehub.dataset_download(self.config.kaggle_dataset_path)
        self.local_save_path = Path(save_path)/self.config.kaggle_dataset_path_suffix
        print(f" File saved in {self.local_save_path}")