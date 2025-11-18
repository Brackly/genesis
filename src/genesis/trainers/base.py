from abc import ABC, abstractmethod
from genesis.metrics import Metric



class BaseTrainer(ABC):
    def __init__(self):
        self.model = None
        self.metrics = list[Metric]

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
    @abstractmethod
    def score_fn(self ,*args ,**kwargs):
        pass
    @abstractmethod
    def eval(self ,*args ,**kwargs):
        pass


