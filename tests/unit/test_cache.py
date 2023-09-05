import json
import os
import shutil
import sys
import unittest
from unittest.mock import patch

import pandas as pd

from frameit.cache import (_generate_cache_hash_key, _load_cache_meta,
                           _save_cache_meta, cache_item, evict, in_cache)


def load_mock_cache_items():
    path = os.path.join(os.getcwd(), "tests", "mock_cache.json")

    with open(path) as f:
        cache = json.load(f)

    return cache


def load_mock_frameit_meta():
    path = os.path.join(r"tests/mock_frameit_meta.json")
    with open(path) as f:
        cache_meta = json.load(f)
    return cache_meta


class TestGenerateHashKey(unittest.TestCase):
    def test_hash_key_generation_for_func_with_single_default_argument(self):
        # ARRANGE
        def single_argument_func(a=12):
            return None

        fargs = []
        fkwargs = {"a": 12}

        expected_hash_key = {
            "mod": "tests.unit.test_cache",
            "fun": "single_argument_func",
            "argsize": sys.getsizeof(fargs)
            + sys.getsizeof(fkwargs)
            + sys.getsizeof(fkwargs["a"]),
            "pkg": "pandas",
            "args": {"a": "12"},
        }

        # ACT
        hash_key = _generate_cache_hash_key(
            single_argument_func, fargs=fargs, fkwargs=fkwargs, pkg="pandas"
        )

        # ASSERT
        self.assertDictEqual(expected_hash_key, hash_key)

    def test_hash_key_truncation_of_passed_argument_values_to_five_hundred_chars(self):
        # ARRANGE
        def dummy(a, b, c, d):
            return None

        a = "[" * 800
        b = "-" * 900
        c = "c" * 192
        d = "d" * 2

        expected_args = {"a": "[" * 500, "b": "-" * 500, "c": "c" * 192, "d": "d" * 2}

        # ACT
        hash_key = _generate_cache_hash_key(
            dummy, fargs=[a, b, c, d], fkwargs={}, pkg="pandas"
        )

        # ASSERT
        self.assertEqual(expected_args, hash_key["args"])

    def test_hash_key_generation_for_dataframe_argument(self):
        # ARRANGE
        def dummy(df):
            return None

        df = pd.DataFrame({"one": [2] * 200, "two": [3] * 200, "three": [1] * 200})

        expected_hash_key = {
            "mod": "tests.unit.test_cache",
            "fun": "dummy",
            "argsize": sys.getsizeof([df]) + sys.getsizeof(df) + sys.getsizeof({}),
            "pkg": "pandas",
            "args": {"df": str(df)[0:500]},
        }

        # ACT
        hash_key = _generate_cache_hash_key(dummy, fargs=[df], fkwargs={}, pkg="pandas")

        # ARRANGE
        self.assertDictEqual(expected_hash_key, hash_key)


class TestGetPath(unittest.TestCase):
    pass


class TestInCache(unittest.TestCase):
    @patch("frameit.cache._load_cache_meta")
    def test_in_cache_for_single_cached_dataframe(self, mock_cache_meta):
        # ARRANGE
        mock_cache_meta.return_value = load_mock_frameit_meta()

        hash_key = {
            "mod": "__main__",
            "fun": "multiply",
            "argsize": "12000",
            "pkg": "pandas",
            "args": {"df": "[/n column1 , column2 /n 0.000, 0.002 /n]", "factor": "24"},
        }

        files = ["5jg3ad3521l.parquet"]

        # ACT
        result = in_cache(".cache/frameit", hash_key)

        # ASSERT
        self.assertTrue(bool(result))
        self.assertEqual(files, result)

    @patch("frameit.cache._load_cache_meta")
    def test_in_cache_for_cached_iterable_dataframes(self, mock_cache_meta):
        # ARRANGE
        mock_cache_meta.return_value = load_mock_frameit_meta()

        hash_key = {
            "mod": "__main__",
            "fun": "split",
            "argsize": "12000",
            "pkg": "pandas",
            "args": {"df": "[/n column1 , column2 /n 0.000, 0.002 /n]", "nframes": "3"},
        }

        files = ["g3g5tgg3521l.parquet", "prrvad35rls.parquet", "badgrfadfa.parquet"]

        # ACT
        result = in_cache(".cache/frameit", hash_key)

        # ASSERT
        self.assertTrue(files)
        self.assertEqual(files, result)

    @patch("frameit.cache._load_cache_meta")
    def test_in_cache_for_uncached_item(self, mock_cache_meta):
        # ARRANGE
        mock_cache_meta.return_value = load_mock_frameit_meta()

        hash_key = {
            "mod": "core.utils",
            "fun": "split",
            "argsize": "12000",
            "pkg": "pandas",
            "args": {"df": "[/n column1 , column2 /n 0.000, 0.002 /n]", "nframes": "3"},
        }

        files = []

        # ACT
        result = in_cache(".cache/frameit", hash_key)

        # ASSERT
        self.assertFalse(result)
        self.assertEqual(result, files)


class TestCacheItem(unittest.TestCase):
    path = "tmp/.cache/frameit/test"

    def setUp(self) -> None:
        os.makedirs(self.path)

    def tearDown(self) -> None:
        shutil.rmtree("tmp")

    def test_item_caching_for_single_pandas_dataframe(self):
        # ARRANGE
        df = pd.DataFrame({"one": [1] * 4, "two": [2] * 4})

        hash_key = {
            "mod": "tests.unit.test_cache",
            "fun": "dummy",
            "argsize": 5028,
            "pkg": "pandas",
            "args": {"data": str(df), "gamma": "128.04", "beta": "-0.00451"},
        }

        # ACT
        cache_item(df, cache_path=self.path, cache_meta=[], hash_key=hash_key)

        # ASSERT
        self.assertEqual(
            len(os.listdir(self.path)), 1 + 1
        )  # 1 parquet file and 1 json file

    def test_item_caching_for_tuple_of_pandas_dataframes(self):
        # ARRANGE
        df1 = pd.DataFrame({"one": [1] * 4, "two": [2] * 4})
        df2 = pd.DataFrame({"one": [2] * 4, "two": [4] * 4})
        df3 = pd.DataFrame({"one": [4] * 4, "two": [8] * 4})

        dfs = (df1, df2, df3)

        hash_key = {
            "mod": "tests.unit.test_cache",
            "fun": "dummy",
            "argsize": 5028,
            "pkg": "pandas",
            "args": {"data": str(dfs), "gamma": "128.04", "beta": "-0.00451"},
        }

        # ACT
        cache_item(dfs, cache_path=self.path, cache_meta=[], hash_key=hash_key)

        # ASSERT
        self.assertEqual(
            len(os.listdir(self.path)), 3 + 1
        )  # 3 parquet files plus the frameit_meta.json file


class TestEvict(unittest.TestCase):
    path = "tmp/.cache/frameit/test"

    def setUp(self) -> None:
        os.makedirs(self.path)

    def tearDown(self) -> None:
        shutil.rmtree("tmp")

    @patch("frameit.cache.get_cache_stats")
    def test_eviction_of_single_dataframe_exceeding_cache_time_limit_in_empty_cache(
        self, mock_get_cache_stats
    ):
        # ARRANGE
        mock_get_cache_stats.return_value = {
            "lifetime": {"test_file": 40},
            "size": {"test_file": 1e4},
            "total_size": 1e4,
        }

        # Create a dummy parquet file with a dataframe
        df = pd.DataFrame({"one": [1] * 5, "two": [2] * 5})
        df.to_parquet(os.path.join(self.path, "test_file"))

        hash_key = {
            "mod": "tests.unit.test_cache",
            "fun": "dummy",
            "argsize": 5028,
            "pkg": "pandas",
            "args": {"data": str([1] * 5), "gamma": "128.04", "beta": "-0.00451"},
            "files": ["test_file"],
        }

        expected_meta = []

        # ACT
        _save_cache_meta(cache_path=self.path, cache_meta=[hash_key])
        evict(incoming_size=1e2, cache_path=self.path, max_time=30, max_size=5e6)
        meta = _load_cache_meta(cache_path=self.path)

        # ASSERT
        self.assertEqual(expected_meta, meta)

        # Verify that the cache has been emptied (only the frameit_cache.json
        # should live in the cache)
        files_in_cache = os.listdir(self.path)

        self.assertEqual(files_in_cache.pop(0), "frameit.json")
        self.assertEqual(len(files_in_cache), 0)  # cache is empty

    @patch("frameit.cache.get_cache_stats")
    def test_eviction_of_multiple_dataframes_exceeding_cache_time_limit_in_empty_cache(
        self, mock_get_cache_stats
    ):
        # ARRANGE

        # Simulate that the files have been living in the cache for 40 minutes
        mock_get_cache_stats.return_value = {
            "lifetime": {"file1": 40, "file2": 40, "file3": 40, "file4": 40},
            "size": {"file1": 1e4, "file2": 1e4, "file3": 1e4, "file4": 1e4},
            "total_size": 4e4,
        }

        # Create a set of parquet files all generated from one method with the same
        # input args. This simulates the case `return df1, df2, df3, df4`
        df = pd.DataFrame({"one": [1] * 5, "two": [2] * 5})

        df.to_parquet(os.path.join(self.path, "file1"))
        df.to_parquet(os.path.join(self.path, "file2"))
        df.to_parquet(os.path.join(self.path, "file3"))
        df.to_parquet(os.path.join(self.path, "file4"))

        hash_key = {
            "mod": "tests.unit.test_cache",
            "fun": "dummy",
            "argsize": 5028,
            "pkg": "pandas",
            "args": {"data": str([1] * 5), "gamma": "128.04", "beta": "-0.00451"},
            "files": ["file1", "file2", "file3", "file4"],
        }

        expected_meta = []

        # ACT
        _save_cache_meta(cache_path=self.path, cache_meta=[hash_key])
        evict(incoming_size=1e2, cache_path=self.path, max_time=30, max_size=5e6)
        meta = _load_cache_meta(cache_path=self.path)

        # ASSERT
        self.assertEqual(expected_meta, meta)

        # Verify that the cache has been emptied (only the frameit.json should live
        # in the cache)
        files_in_cache = os.listdir(self.path)

        self.assertEqual(files_in_cache.pop(0), "frameit.json")
        self.assertEqual(len(files_in_cache), 0)  # cache is empty

    @patch("frameit.cache.get_cache_stats")
    def test_eviction_of_single_dataframe_exceeding_cache_limit_in_populated_cache(
        self, mock_get_cache_stats
    ):
        # ARRANGE
        # Create a set of parquet files all generated from one method with the same
        # input args. This simulates the case `return df1, df2, df3, df4`
        df = pd.DataFrame({"one": [1] * 5, "two": [2] * 5})

        df.to_parquet(os.path.join(self.path, "file1"))
        df.to_parquet(os.path.join(self.path, "file2"))
        df.to_parquet(os.path.join(self.path, "file3"))
        df.to_parquet(os.path.join(self.path, "file4"))

        # Simulate that the files have been living in the cache for 40 minutes
        mock_get_cache_stats.return_value = {
            "lifetime": {"file1": 10, "file2": 10, "file3": 40, "file4": 10},
            "size": {"file1": 1e4, "file2": 1e4, "file3": 1e4, "file4": 1e4},
            "total_size": 4e4,
        }

        hash_key = [
            {
                "mod": "module.tools",
                "fun": "compute",
                "argsize": 1.1e4,
                "pkg": "pandas",
                "args": {},
                "files": ["file1"],
            },
            {
                "mod": "module.tools",
                "fun": "eval",
                "argsize": 1.2e4,
                "pkg": "pandas",
                "args": {},
                "files": ["file2"],
            },
            {
                "mod": "module.tools",
                "fun": "__new__",
                "argsize": 1.2e4,
                "pkg": "pandas",
                "args": {},
                "files": ["file3"],
            },
            {
                "mod": "module.tools",
                "fun": "__new__",
                "argsize": 1.2e4,
                "pkg": "pandas",
                "args": {},
                "files": ["file4"],
            },
        ]

        expected_meta = hash_key.copy()
        expected_meta.pop(2)

        # ACT
        _save_cache_meta(cache_path=self.path, cache_meta=hash_key)
        evict(incoming_size=1e3, cache_path=self.path, max_size=5e5, max_time=30)
        meta = _load_cache_meta(cache_path=self.path)

        # ASSERT
        self.assertEqual(meta, expected_meta)
