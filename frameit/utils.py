import copy
import inspect
import os
from time import time
from typing import Any, Dict


def parse_package_from_type(item_type: type) -> str:
    return item_type.__module__.split(".").pop(0)


def getargspec(func, fargs: tuple, fkwargs: dict) -> Dict[str, Any]:
    flargs = list(fargs)

    sig = inspect.signature(func)
    params = sig.parameters

    argspec = dict()

    for name in params.keys():
        if name == "kwargs":
            continue
        if name in fkwargs.keys():
            value = fkwargs.pop(name)
        elif flargs:
            value = flargs.pop(0)
        else:
            value = params[name].default
        argspec[name] = value

    assert len(flargs) == 0
    assert len(fkwargs) == 0

    return argspec


def get_cache_stats(path) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}

    stats["lifetime"] = {}
    stats["size"] = {}
    stats["total_size"] = 0

    # If the cache directory does not exist no items have been cached, so we return the
    # original cache stats dictionary.
    if not os.path.isdir(path):
        return stats

    files = os.listdir(path)

    for file in files:
        # If the dataframe is chunked as an iterable, extract the stats from all
        # individual chunks
        item_path = os.path.join(path, file)

        if file == "frameit.json":
            continue

        memsize = os.path.getsize(item_path) / 1000
        lifetime = (time() - os.path.getmtime(item_path)) / 60
        stats["lifetime"][file] = lifetime
        stats["size"][file] = memsize
        stats["total_size"] += memsize

    return stats


def deepcopy(obj, memo=None):
    if memo is None:
        memo = {}

    if id(obj) in memo:
        return memo[id(obj)]

    if isinstance(obj, (int, str, float, bool)):
        # For basic types, return a deep copy
        return copy.deepcopy(obj)
    elif isinstance(obj, list):
        # Create a new list and add deep copies of its elements
        new_list = []
        memo[id(obj)] = new_list
        for item in obj:
            new_list.append(deepcopy(item, memo))
        return new_list
    elif isinstance(obj, dict):
        # Create a new dictionary and add deep copies of its items
        new_dict = {}
        memo[id(obj)] = new_dict
        for key, value in obj.items():
            new_key = deepcopy(key, memo)
            new_value = deepcopy(value, memo)
            new_dict[new_key] = new_value
        return new_dict
    else:
        # For other types, return a shallow copy
        return copy.copy(obj)
