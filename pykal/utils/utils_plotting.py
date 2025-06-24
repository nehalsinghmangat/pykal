import inspect
from typing import Callable, Set, Optional, get_type_hints, get_origin, get_args, Union
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import List, Tuple, Union
from functools import wraps
from utils.base_utils import *


def combine_data_and_time_into_DataFrame(
    data: NDArray, time: NDArray, column_names: List[str]
) -> pd.DataFrame:
    """
    Combine a 2D data array and a 1D time array into a pandas DataFrame.

    Parameters
    ----------
    data : NDArray
        Array of shape (n_variables, n_time_steps).
    time : NDArray
        Array of shape (n_time_steps,) or (n_time_steps, 1).
    column_names : list
        List of length n_variables giving column names for the data.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame with time as the index and variable names as columns.

    Raises
    ------
    ValueError
        If shapes of data, time, or column_names are incompatible.
    TypeError
        If input types are incorrect.

    Examples
    --------
    >>> import numpy as np
    >>> import System
    >>> data = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    >>> time = np.array([0.1, 0.2, 0.3])
    >>> column_names = ["x0", "x1"]
    >>> df = System.combine_data_and_time_into_DataFrame(data, time, column_names)
    >>> df
          x0    x1
    time
    0.1   1.0  10.0
    0.2   2.0  20.0
    0.3   3.0  30.0

    >>> bad_data = np.array([1.0, 2.0, 3.0])
    >>> System.combine_data_and_time_into_DataFrame(bad_data, time, column_names)
    Traceback (most recent call last):
        ...
    ValueError: `data` must be a 2D array, got shape (3,).

    >>> time_wrong = np.array([0.1, 0.2])
    >>> System.combine_data_and_time_into_DataFrame(data, time_wrong, column_names)
    Traceback (most recent call last):
        ...
    ValueError: Time length 2 does not match number of time steps in data 3.

    >>> bad_names = ["x0"]
    >>> System.combine_data_and_time_into_DataFrame(data, time, bad_names)
    Traceback (most recent call last):
        ...
    ValueError: Length of column_names (1) does not match number of variables in data (2).

    >>> not_a_list = "x0,x1"
    >>> System.combine_data_and_time_into_DataFrame(data, time, not_a_list)
    Traceback (most recent call last):
        ...
    TypeError: `column_names` must be a list, got <class 'str'>

    >>> time_matrix = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    >>> System.combine_data_and_time_into_DataFrame(data, time_matrix, column_names)
    Traceback (most recent call last):
        ...
    ValueError: `time` should be of shape (n_steps,) or (n_steps, 1), got (2, 3).

    >>> System.combine_data_and_time_into_DataFrame(data, time, tuple(column_names))
    Traceback (most recent call last):
        ...
    TypeError: `column_names` must be a list, got <class 'tuple'>

    """

    if not isinstance(data, np.ndarray):
        raise TypeError(f"`data` must be an np.ndarray, got {type(data)}")
    if not isinstance(time, np.ndarray):
        raise TypeError(f"`time` must be an np.ndarray, got {type(time)}")
    if not isinstance(column_names, list):
        raise TypeError(f"`column_names` must be a list, got {type(column_names)}")

    if data.ndim != 2:
        raise ValueError(f"`data` must be a 2D array, got shape {data.shape}.")

    n_vars, n_steps = data.shape

    if time.ndim == 2:
        if time.shape[1] != 1:
            raise ValueError(
                f"`time` should be of shape (n_steps,) or (n_steps, 1), got {time.shape}."
            )
        time = time.flatten()
    elif time.ndim != 1:
        raise ValueError(
            f"`time` must be 1D or 2D with one column, got shape {time.shape}."
        )

    if time.shape[0] != n_steps:
        raise ValueError(
            f"Time length {time.shape[0]} does not match number of time steps in data {n_steps}."
        )

    if len(column_names) != n_vars:
        raise ValueError(
            f"Length of column_names ({len(column_names)}) does not match number of variables in data ({n_vars})."
        )

    df = pd.DataFrame(data.T, index=time, columns=column_names)
    df.index.name = "time"
    return df
