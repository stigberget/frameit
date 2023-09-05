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
