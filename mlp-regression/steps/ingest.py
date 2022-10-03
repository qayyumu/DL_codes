"""
This module defines the following routines used by the 'ingest' step of the regression pipeline:

- ``load_file_as_dataframe``: Defines customizable logic for parsing dataset formats that are not
  natively parsed by MLflow Pipelines (i.e. formats other than Parquet, Delta, and Spark SQL).
"""

import logging

from pandas import DataFrame

_logger = logging.getLogger(__name__)


def load_file_as_dataframe(file_path: str, file_format: str) -> DataFrame:


    if file_format == "csv":
        import pandas

        return pandas.read_csv(file_path)
    else:
        raise NotImplementedError


def load_dataset_from_tfds(dataset:str='radon', file_format:str='dataframe', file_path:str=None) -> DataFrame:
    
    import tensorflow_datasets as tfds
    
    _logger.info(
            "Loading radon dataset from tensorflow datasets, output format will be a pandas dataframe"
        )
    
    ds = tfds.load(dataset, split='train')
    df = tfds.as_dataframe(ds)
    
    return df