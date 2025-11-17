from abc import ABC, abstractmethod


class Fetcher(ABC):
    def __init__(self):
        self.local_save_path = None
    @abstractmethod
    def fetch(self):
        pass