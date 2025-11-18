from src.writers import base,tensorboard,wandb,composite
from typing import  Union
from pathlib import Path

def writer_factory(writer_type: str,
                  log_dir: Union[str, Path],
                  experiment_name: str = None,
                  **kwargs) -> base.BaseWriter:
    """
    Factory function to create writers

    Args:
        writer_type: One of 'tensorboard', 'wandb', 'noop', or 'composite'
        log_dir: Directory for logging
        experiment_name: Name of the experiment
        **kwargs: Additional arguments for specific writers

    Returns:
        Instance of BaseWriter
    """
    writers = {
        'tensorboard': tensorboard.TensorBoardWriter,
        'wandb': wandb.WandBWriter,
        'noop': base.NoOpWriter
    }

    if writer_type == 'composite':
        # Create multiple writers
        writer_list = []
        for wtype in kwargs.get('writers', ['tensorboard']):
            writer_list.append(writers[wtype](log_dir, experiment_name))
        return composite.CompositeWriter(writer_list)

    if writer_type not in writers:
        raise ValueError(f"Unknown writer type: {writer_type}. Choose from {list(writers.keys())}")

    return writers[writer_type](log_dir, experiment_name, **kwargs)
