import json
import os
import platform
import shutil
import uuid
from functools import wraps
from logging import Logger
from sys import getsizeof
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union, get_args

import pandas as pd
import polars as pl

from frameit.utils import (
    deepcopy,
    get_cache_stats,
    getargspec,
    parse_package_from_type
)

CHAR_LIMIT = 500
TIME_LIMIT = 60
CACHE_LIMIT = 16e6  # = 16 GB (base in kB)

FRAME_OR_SERIES = Union[pd.DataFrame, pd.Series, pl.DataFrame, pl.LazyFrame, pl.Series]

CACHEDIR = "frameit"

logger = Logger(name="frameit")


def _get_cache_path() -> str:
    if platform.system() == "Darwin":
        path = "/Library/Caches"
    elif platform.system() == "Windows":
        raise NotImplementedError()
    elif platform.system() == "Linux":
        raise NotImplementedError()

    path = os.path.join(path, CACHEDIR)

    return path


def _load_cache_meta(cache_path: str) -> List[Dict[str, Any]]:
    """
    Loads cache metadata into memory.

    Parameters
    ----------
    cache_path: str
        Path to the frameit cache.

    Returns
    -------
    List[Dict[str, Any]]
        Cache metadata associating unique hash keys to a set of files in the cache.
    """
    cache_meta_path = os.path.join(cache_path, "frameit.json")
    if not os.path.isfile(cache_meta_path):
        return []

    with open(cache_meta_path) as f:
        meta = json.load(f)

    if not meta:
        return []

    return meta


def _save_cache_meta(cache_path: str, cache_meta: List[Dict[str, Any]]) -> None:
    """
    Saves cache metadata to memory (in json format)

    Parameters
    ----------
    cache_path: str
        Path to the frameit cache.
    cache_meta: List[Dict[str, Any]]
        Cache metadata.
    """

    cache_meta_path = os.path.join(cache_path, "frameit.json")
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    with open(cache_meta_path, mode="w") as f:
        json.dump(cache_meta, f)


def _generate_cache_hash_key(
    func: Callable, fargs: tuple, fkwargs: dict, pkg: str
) -> Dict[str, Any]:
    """
    Generates a unique hash key that is determined by the function name, where the
    function is defined, and the function arguments passed to the function.

    Parameters
    ----------
    func:
        The decorated function.
    fargs:
        Function argument values.
    fkwargs:
        Function keyword argument values.
    pkg:
        The type of the DataFrame returned from `func`.

    Returns
    ----------
    Dict[str, Any]:
        The unique hash key.
    """
    args_size = sum([getsizeof(fargs)] + [getsizeof(farg) for farg in fargs]) + sum(
        [getsizeof(fkwargs)] + [getsizeof(value) for value in fkwargs.values()]
    )
    argspec = getargspec(func, fargs, fkwargs)

    hash_key = {
        "mod": func.__module__,
        "fun": func.__name__,
        "argsize": args_size,
        "pkg": pkg,
        "args": {arg: str(value)[0:CHAR_LIMIT] for arg, value in argspec.items()},
    }

    return hash_key


def evict(
    incoming_size: float, cache_path: str, max_size: float, max_time: float
) -> List[Dict[str, Any]]:
    """
    Removes items in the file cache following the LRU approach. An item is evicted if its
    lifetime in the cache exceeds the maximum allowable time (`max_time`) of an item, or
    if the total size of the cache is exceeded.

    Parameters
    ----------
    incoming_size: float
        The size (in kB) of the incoming item
    cache_path: str
        Path to the frameit cache.
    max_size: float
        Maximum set size (in kB) of the cache.
    max_time: float
        Maximum residence time (in min).

    Returns
    -------
    List[Dict[str, Any]]
        An updated version of the cache metadata.
    """
    stats = get_cache_stats(cache_path)
    meta = _load_cache_meta(cache_path)
    metamap = {index: metadata_item for index, metadata_item in enumerate(meta)}

    lifetime = {
        k: v
        for k, v in sorted(
            stats["lifetime"].items(), key=lambda item: item[1], reverse=True
        )
    }

    # Get the files into a more readable format
    file_list = [meta_item["files"] for meta_item in metamap.values()]
    file_mapping = {
        file: index for index, files in enumerate(file_list) for file in files
    }

    # Initialize the files dictionary with the corresponding index hash_key id
    files = list(lifetime.keys())

    # Iterate over the files in the cache (these are sorted according to their lifespan in
    # the cache)
    while files:

        file = files[0]

        if (
            lifetime[file] < max_time
            and (stats["total_size"] + incoming_size) < max_size
        ):
            break

        index = file_mapping[file]
        metamap.pop(index)

        for associated_files in file_list[index]:
            stats["lifetime"].pop(associated_files)
            item_size = stats["size"].pop(associated_files)
            stats["total_size"] -= item_size
            filepath = os.path.join(cache_path, associated_files)
            files.remove(associated_files)
            os.remove(filepath)

    updated_cache_meta = [metamap[index] for index in metamap]
    _save_cache_meta(cache_path=cache_path, cache_meta=updated_cache_meta)
    return updated_cache_meta


def clear_cache(cache_path="./.cache/frameit") -> None:
    """
    Removes any items cached in the `cache_path`.

    Parameters
    ----------
    cache_path: Union[str, BytesIO]
        Path to the frameit cache.
    """
    if os.path.isdir(cache_path):
        shutil.rmtree(cache_path)


def in_cache(cache_path: str, hash_key: Dict[str, Any]) -> List[str]:
    """
    Checks whether a specific `hash_key` exists in the `cache_path`.

    Parameters
    ----------
    cache_path: str
        Path to the frameit cache.
    hash_key: dict
        A unique identifier that is determined by the function name, where the
        function is defined, and the function arguments passed to the function.

    Returns
    ----------
    List[str]:
        A list of the files that correspond to the given `hash_key`. If the `hash_key`
        does not exist in the cache, an empty `list` is returned.
    """
    metadata = _load_cache_meta(cache_path)

    for metakey in metadata:
        files = metakey.pop("files")
        if metakey == hash_key:
            return files

    return []


def cache_item(
    dfs: FRAME_OR_SERIES,
    cache_path: str,
    cache_meta: List[Dict[str, Any]],
    hash_key: Dict[str, Any],
) -> None:
    """
    Saves a single or set of DataFrame(s) to the cache, and associates the
    DataFrame(s) with their `hash_key`.

    Parameters
    ----------
    dfs: FRAME_OR_SERIES
        A collection of pandas or polars `DataFrame` or `Series`.
    cache_path: str
        Path to the frameit cache.
    cache_meta:
        The full cache hash_key store.
    hash_key: Dict[str, Any]
        A unique identifier that is determined by the function name, where the
        function is defined, and the function arguments passed to the function.
    """

    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)
        cache_meta = []
    else:
        cache_meta = _load_cache_meta(cache_path)

    TYPES = get_args(FRAME_OR_SERIES)

    if isinstance(dfs, TYPES):
        file = uuid.uuid4().hex
        item_path = os.path.join(cache_path, file)

        if hash_key["pkg"] == "pandas":
            dfs.to_parquet(item_path)
        elif hash_key["pkg"] == "polars":
            dfs.write_parquet(item_path)
        else:
            raise KeyError(
                f"The given pkg key is not recognized. "
                f"Permissible keys ['pandas', 'polars']. "
                f"Received {hash_key['pkg']}"
            )
        files = [file]
    else:
        files = []
        for df in dfs:
            file = uuid.uuid4().hex
            item_path = os.path.join(cache_path, file)
            if hash_key["pkg"] == "pandas":
                df.to_parquet(item_path)
            elif hash_key["pkg"] == "polars":
                df.write_parquet(item_path)
            else:
                raise KeyError(
                    f"The given pkg key is not recognized. "
                    f"Permissible keys ['pandas', 'polars']. "
                    f"Received {hash_key['pkg']}"
                )
            files.append(file)

    hash_key["files"] = files
    cache_meta.append(hash_key)

    _save_cache_meta(cache_path, cache_meta)


def get_size(dfs: FRAME_OR_SERIES) -> int:
    """
    Evaluates the size of the items being cached.

    Parameters
    ----------
    dfs: FRAME_OR_SERIES
        A collection of pandas or polars `DataFrame` or `Series`.

    Returns
    ----------
    int:
        The collective size of the pandas or polars objects.

    """
    if isinstance(dfs, get_args(FRAME_OR_SERIES)):
        try:
            frame_size = dfs.memory_usage(index=True).sum() / 1000
        except:
            frame_size = dfs.estimated_size("kb")
    else:
        # CODE BLOCK FOR CHUNKED DATAFRAME
        frame_size = 0
        for df in dfs:
            try:
                item_size = (
                    df.memory_usage(index=True).sum() / 1000
                )  # Estimated size from pd.DataFrame
            except:
                item_size = df.estimated_size("kb")  # Estimated size from pl.DataFrame

            frame_size += item_size

    return frame_size


def _verify_hashkey(
    cache_path: str, func: Callable, fargs: tuple, fkwargs: dict
) -> Tuple[List[str], str]:
    """
    Checks whether the `hash_key` for a unique set of arguments, exists
    in the cache.

    Parameters
    ----------
    cache_path: str
        The path to the frameit cache.
    func: Callable
        The functiion callable calling the function.
    fargs: Tuple
        A tuple containing the arguments passed to `func`.
    fkwargs: Dict[str, Any]
        A dictionary containing the keyword arguments passed to `func`.

    Returns
    -------
    Tuple[List[str], str]
        A list of files associated with a given `hash_key`, and the associated
        DataFrame library generating the DataFrame.

    """

    hash_key_pandas = _generate_cache_hash_key(
        func, fargs=fargs, fkwargs=fkwargs.copy(), pkg="pandas"
    )
    hash_key_polars = _generate_cache_hash_key(
        func, fargs=fargs, fkwargs=fkwargs.copy(), pkg="polars"
    )

    pandas_objects = in_cache(cache_path, hash_key_pandas)
    polars_objects = in_cache(cache_path, hash_key_polars)

    if pandas_objects:
        return pandas_objects, "pandas"
    elif polars_objects:
        return polars_objects, "polars"
    else:
        return [], ""


def frameit(
    func=None,
    max_time=10,
    max_size=1e4,
    cache_dir="./.cache/frameit",
    itertype=tuple,
) -> FRAME_OR_SERIES:
    def wrapper(func):
        @wraps(func)
        def cache(*args, **kwargs):
            fargs = deepcopy(args)
            fkwargs = deepcopy(kwargs)

            if max_size > CACHE_LIMIT:
                raise ValueError(
                    f"The cache limit is not allowed to exceed {CACHE_LIMIT} [kB]"
                )
            elif max_time > TIME_LIMIT:
                raise ValueError(
                    f"The time limit is not allowed to exceed {TIME_LIMIT} [min]"
                )
            cache_path = os.path.abspath(cache_dir) if cache_dir else _get_cache_path()
            evict(0.0, cache_path, max_size, max_time)  # Remove any expired items

            cached_items, pkg = _verify_hashkey(
                cache_path, func, fargs=fargs, fkwargs=fkwargs
            )

            if cached_items:
                if len(cached_items) > 1:
                    dfs = []
                    for cached_item in cached_items:
                        filepath = os.path.join(cache_dir, cached_item)
                        if pkg == "pandas":
                            dfs.append(pd.read_parquet(filepath))
                        elif pkg == "polars":
                            dfs.append(pl.read_parquet(filepath))
                    return itertype(dfs)
                else:
                    filepath = os.path.join(cache_dir, cached_items[0])
                    if pkg == "pandas":
                        return pd.read_parquet(filepath)
                    elif pkg == "polars":
                        return pl.read_parquet(filepath)

            rvs = func(*args, **kwargs)

            TYPES = get_args(FRAME_OR_SERIES)
            if not isinstance(rvs, TYPES) and isinstance(rvs, Iterable):
                item_types = []
                for item in rvs:
                    item_types.append(isinstance(item, TYPES))

                if not all(item_types):
                    return rvs

                pkg = parse_package_from_type(item_type=type(rvs[0]))
            elif not isinstance(rvs, TYPES):
                return rvs
            else:
                pkg = parse_package_from_type(item_type=type(rvs))

            dfs = rvs

            item_size = get_size(dfs)
            hash_key = _generate_cache_hash_key(
                func=func, fargs=fargs, fkwargs=fkwargs, pkg=pkg
            )

            if item_size <= CACHE_LIMIT:
                cache_meta = evict(item_size, cache_path, max_size, max_time)
                cache_item(dfs, cache_path, cache_meta, hash_key)
            else:
                logger.debug("The size of the dataframe is too large to be cached")

            return dfs

        return cache

    # Handle other cases, e.g., @mydecorator(func=myfunc)
    if func is None:
        return wrapper
    elif callable(func):
        return wrapper(func)
    else:
        raise ValueError(
            "Invalid call to frameit. Frameit only handles the decorated function"
        )
