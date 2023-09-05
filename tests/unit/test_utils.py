import json
import os
import random
import shutil
import unittest
from unittest.mock import patch

import pandas as pd

from frameit.utils import get_cache_stats, getargspec


class TestGetCacheStats(unittest.TestCase):
    def get_mock_frame(self):
        return pd.DataFrame([[random.random()] * 5, [random.random()] * 5])

    def setUp(self) -> None:
        self.cache = "tmp/.cache/frameit/test"
        os.makedirs(self.cache)
        self.file = os.path.join(self.cache, "cachedfile")

    def tearDown(self):
        if os.path.exists(self.cache):
            shutil.rmtree(self.cache)

    def test_cache_meta_json_file_is_ignored_from_cache_stats(self):
        # ARRANGE
        df = self.get_mock_frame()
        df.to_parquet(self.file)

        with open(os.path.join(self.cache, "frameit.json"), mode="w") as f:
            json.dump({}, f)

        # ACT
        stats = get_cache_stats(self.cache)

        # ASSERT
        self.assertTrue("frameit.json" not in stats["lifetime"].keys())
        self.assertTrue("frameit.json" not in stats["size"].keys())

    @patch("frameit.utils.time")
    @patch("os.listdir")
    @patch("os.path.getsize")
    @patch("os.path.getmtime")
    def test_get_cache_stats_for_single_item_in_cache(
        self, mock_getmtime, mock_get_size, mock_list_dir, mock_time
    ):
        # ARRANGE
        creation_time = 15392350.0
        mock_getmtime.return_value = creation_time
        mock_get_size.return_value = 1e6
        mock_list_dir.return_value = ["snappy"]
        mock_time.return_value = creation_time + 5 * 60  # The file has lived for 5 min

        expected = {
            "lifetime": {"snappy": 5.0},
            "size": {"snappy": 1e3},
            "total_size": 1e3,
        }

        # ACT
        result = get_cache_stats(self.cache)

        # ASSERT
        self.assertDictEqual(result, expected)

    @patch("frameit.utils.time")
    @patch("os.listdir")
    @patch("os.path.getsize")
    @patch("os.path.getmtime")
    def test_get_cache_stats_for_cached_iterable_in_cache(
        self,
        mock_getmtime,
        mock_get_size,
        mock_list_dir,
        mock_time,
    ):
        # ARRANGE
        creation_time = 15392350.0
        mock_getmtime.return_value = creation_time
        mock_get_size.return_value = 1e7
        mock_list_dir.return_value = ["snappy1", "snappy2", "snappy3", "snappy4"]
        mock_time.return_value = creation_time + 10 * 60

        expected = {
            "lifetime": {
                "snappy1": 10.0,
                "snappy2": 10.0,
                "snappy3": 10.0,
                "snappy4": 10.0,
            },
            "size": {"snappy1": 1e4, "snappy2": 1e4, "snappy3": 1e4, "snappy4": 1e4},
            "total_size": 4e4,
        }

        # ACT
        result = get_cache_stats(self.cache)

        # ARRANGE
        self.assertDictEqual(expected, result)


class TestGetargspecs(unittest.TestCase):
    @staticmethod
    def dummy_with_no_defaults(a, b, c):
        return a * b * c

    @staticmethod
    def dummy_with_defaults(a, b, c=2, d="mockit"):
        return str(a * b * c) + d

    def test_correct_argspecs_created_for_fun_with_no_defaults_and_no_kwargs_passed(
        self,
    ):
        # ARRANGE (& ACT)
        expected_argspec = {
            "a": 1,
            "b": 2,
            "c": 3,
        }

        argspec = getargspec(self.dummy_with_no_defaults, fargs=[1, 2, 3], fkwargs={})

        # ASSERT
        self.assertDictEqual(expected_argspec, argspec)

    def test_correct_argspecs_created_for_fun_with_no_defaults_and_single_kwarg_passed(
        self,
    ):
        # ARRANGE (& ACT)
        expected_argspec = {
            "a": 1,
            "b": 2,
            "c": 3,
        }

        argspec = getargspec(self.dummy_with_no_defaults, fargs=[1, 2], fkwargs={"c": 3})

        # ASSERT
        self.assertDictEqual(expected_argspec, argspec)

    def test_correct_argspecs_created_for_fun_with_no_defaults_and_multiple_kwargs_passed(
        self,
    ):
        # ARRANGE (& ACT)
        expected_argspec = {"a": 1, "b": 2, "c": 3}

        argspec = getargspec(
            self.dummy_with_no_defaults, fargs=[1], fkwargs={"b": 2, "c": 3}
        )

        # ASSERT
        self.assertDictEqual(expected_argspec, argspec)

    def test_correct_argspecs_created_for_fun_with_no_defaults_defined_fully_with_kwargs(
        self,
    ):
        # ARRANGE (& ACT)
        expected_argspec = {
            "a": 1,
            "b": 2,
            "c": 3,
        }

        argspec = getargspec(
            self.dummy_with_no_defaults, fargs=[], fkwargs={"a": 1, "b": 2, "c": 3}
        )

        # ASSERT
        self.assertDictEqual(expected_argspec, argspec)

    def test_correct_argspecs_created_for_fun_with_defaults_defined_fully_with_args(
        self,
    ):
        # ARRANGE (& ACT)
        expected_argspec = {"a": 1, "b": 2, "c": 3, "d": "id"}

        argspec = getargspec(self.dummy_with_defaults, fargs=[1, 2, 3, "id"], fkwargs={})

        # ASSERT
        self.assertDictEqual(expected_argspec, argspec)

    def test_correct_argspecs_created_for_fun_with_defaults_defined_with_default(self):
        # ARRANGE (& ACT)
        expected_argspec = {"a": 1, "b": 2, "c": 5, "d": "mockit"}

        argspec = getargspec(self.dummy_with_defaults, fargs=(1, 2, 5), fkwargs={})

        # ASSERT
        self.assertDictEqual(expected_argspec, argspec)

    def test_correct_argpsecs_created_for_fun_with_defaults_defined_with_kwargs(self):
        # ARRANGE (& ACT)
        expected_argspec = {"a": 2, "b": 4, "c": 6, "d": "eight"}

        argspec = getargspec(
            self.dummy_with_defaults, fargs=[2, 4], fkwargs={"c": 6, "d": "eight"}
        )

        # ASSERT
        self.assertDictEqual(expected_argspec, argspec)

    def test_correct_argspecs_created_for_fun_with_defaults_defined_fully_with_kwargs(
        self,
    ):
        # ARRANGE (& ACT)
        expected_argspec = {"a": 2, "b": 4, "c": 6, "d": "eight"}

        argspec = getargspec(
            self.dummy_with_defaults,
            fargs=[],
            fkwargs={"a": 2, "b": 4, "c": 6, "d": "eight"},
        )

        # ASSERT
        self.assertDictEqual(expected_argspec, argspec)
