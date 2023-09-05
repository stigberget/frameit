import json
import os
import shutil
import unittest
from unittest.mock import patch

import pandas as pd
import polars as pl

from frameit import frameit

PATH = "./tmp/.cache/frameit"


def inspect_cache() -> list:
    with open(f"{PATH}/frameit.json", "r") as f:
        metadata = json.load(f)
    return metadata


class TestIntegrationForStandardArguments(unittest.TestCase):
    def pandas_read_parquet(self, file):
        return pd.read_parquet(file)

    def polars_read_parquet(self, filename):
        return pl.read_parquet(filename)

    @frameit(cache_dir=PATH)
    def dummy(self, string, integer, dftype):

        if dftype == "polars":
            df = self.pldf * integer
        else:
            df = self.pddf * integer

        return df

    @frameit(cache_dir=PATH)
    def iterable_dummy(self, string, integers, dftype):
        if dftype == "pandas":
            return tuple([self.pddf * integer for integer in integers])
        else:
            return tuple((self.pldf * integer for integer in integers))

    def setUp(self) -> None:
        self.pldf = pl.DataFrame({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
        self.pddf = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})

    def tearDown(self) -> None:
        if os.path.exists("tmp"):
            shutil.rmtree("tmp")

    @patch("pandas.read_parquet", wraps=pd.read_parquet)
    def test_integration_for_pandas_return_type_with_standard_arguments(
        self, mock_pandas_read_parquet
    ):
        # ARRANGE
        argself = (
            "test_integration_for_pandas_return_type_with_standard_arguments "
            + "(tests.integration.test_integration_frameit."
            + "TestIntegrationForStandardArguments)"
        )

        # ACT
        expected = self.dummy("testing", 42, "pandas")
        result = self.dummy("testing", 42, "pandas")

        cache = inspect_cache().pop(0)
        args = cache["args"]

        # ASSERT
        pd.testing.assert_frame_equal(expected, result)

        # Verify that we are loading the cached pandas DataFrame into memory from the
        # correct location
        mock_pandas_read_parquet.assert_called_once_with(f"{PATH}/{cache['files'][0]}")

        self.assertEqual(cache["mod"], "tests.integration.test_integration_frameit")
        self.assertEqual(cache["fun"], "dummy")
        self.assertEqual(cache["pkg"], "pandas")
        self.assertEqual(args["self"], argself)
        self.assertEqual(args["string"], "testing")
        self.assertEqual(args["integer"], "42")
        self.assertEqual(args["dftype"], "pandas")

    @patch("polars.read_parquet", wraps=pl.read_parquet)
    def test_integration_for_polars_return_type_with_standard_arguments(
        self, mock_polars_read_parquet
    ):
        # ARRANGE
        argself = (
            "test_integration_for_polars_return_type_with_standard_arguments "
            + "(tests.integration.test_integration_frameit."
            + "TestIntegrationForStandardArguments)"
        )

        # ACT
        expected = self.dummy("testing", 42, "polars")
        result = self.dummy("testing", 42, "polars")

        cache = inspect_cache().pop(0)
        args = cache["args"]

        # ASSERT
        self.assertTrue(expected.frame_equal(result))

        # Verify that we are loading the cached pandas DataFrame into memory from the
        # correct location
        mock_polars_read_parquet.assert_called_once_with(f"{PATH}/{cache['files'][0]}")

        self.assertEqual(cache["mod"], "tests.integration.test_integration_frameit")
        self.assertEqual(cache["fun"], "dummy")
        self.assertEqual(cache["pkg"], "polars")
        self.assertEqual(args["self"], argself)
        self.assertEqual(args["string"], "testing")
        self.assertEqual(args["integer"], "42")
        self.assertEqual(args["dftype"], "polars")

    @patch("pandas.read_parquet", wraps=pd.read_parquet)
    def test_integration_for_iterable_pandas_return_type_with_standard_arguments(
        self, mock_pandas_read_parquet
    ):
        # ARRANGE
        argself = (
            "test_integration_for_iterable_pandas_return_type_with_standard_arguments "
            + "(tests.integration.test_integration_frameit."
            + "TestIntegrationForStandardArguments)"
        )

        # ACT
        expectations = self.iterable_dummy(
            string="testing", integers=[1, 2, 3, 4], dftype="pandas"
        )
        results = self.iterable_dummy(
            string="testing", integers=[1, 2, 3, 4], dftype="pandas"
        )

        cache = inspect_cache().pop(0)
        args = cache["args"]

        # ASSERT
        for result, expected in zip(results, expectations):
            pd.testing.assert_frame_equal(result, expected)

        mock_pandas_read_parquet.assert_any_call(f"{PATH}/{cache['files'][0]}")
        mock_pandas_read_parquet.assert_any_call(f"{PATH}/{cache['files'][1]}")
        mock_pandas_read_parquet.assert_any_call(f"{PATH}/{cache['files'][2]}")
        mock_pandas_read_parquet.assert_any_call(f"{PATH}/{cache['files'][3]}")

        self.assertEqual(cache["mod"], "tests.integration.test_integration_frameit")
        self.assertEqual(cache["fun"], "iterable_dummy")
        self.assertEqual(cache["pkg"], "pandas")
        self.assertEqual(args["self"], argself)
        self.assertEqual(args["string"], "testing")
        self.assertEqual(args["integers"], "[1, 2, 3, 4]")
        self.assertEqual(args["dftype"], "pandas")

    @patch("polars.read_parquet", wraps=pl.read_parquet)
    def test_integration_for_iterable_polars_return_type_with_standard_arguments(
        self, mock_polars_read_parquet
    ):
        # ARRANGE
        argself = (
            "test_integration_for_iterable_polars_return_type_with_standard_arguments "
            + "(tests.integration.test_integration_frameit."
            + "TestIntegrationForStandardArguments)"
        )

        # ACT
        expectations = self.iterable_dummy(
            string="testing", integers=[1, 2, 3, 4], dftype="polars"
        )
        results = self.iterable_dummy(
            string="testing", integers=[1, 2, 3, 4], dftype="polars"
        )

        cache = inspect_cache().pop(0)
        args = cache["args"]

        # ASSERT
        for result, expected in zip(results, expectations):
            self.assertTrue(result.frame_equal(expected))

        mock_polars_read_parquet.assert_any_call(f"{PATH}/{cache['files'][0]}")
        mock_polars_read_parquet.assert_any_call(f"{PATH}/{cache['files'][1]}")
        mock_polars_read_parquet.assert_any_call(f"{PATH}/{cache['files'][2]}")
        mock_polars_read_parquet.assert_any_call(f"{PATH}/{cache['files'][3]}")

        self.assertEqual(cache["mod"], "tests.integration.test_integration_frameit")
        self.assertEqual(cache["fun"], "iterable_dummy")
        self.assertEqual(cache["pkg"], "polars")
        self.assertEqual(args["self"], argself)
        self.assertEqual(args["string"], "testing")
        self.assertEqual(args["integers"], "[1, 2, 3, 4]")
        self.assertEqual(args["dftype"], "polars")


class TestIntegrationForDataFrameArgument(unittest.TestCase):
    @frameit(cache_dir=PATH)
    def dummy(self, df, multiplier):
        return df * multiplier

    @frameit(cache_dir=PATH)
    def iterable_dummy(self, df, multipliers):
        return tuple([df * multiplier for multiplier in multipliers])

    def setUp(self) -> None:
        self.pldf = pl.DataFrame({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
        self.pddf = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})

    def tearDown(self) -> None:
        if os.path.exists("tmp"):
            shutil.rmtree("tmp")

    def test_full_integration_for_polars_return_type_with_dataframe_argument(self):
        # ARRANGE
        argself = (
            "test_full_integration_for_polars_return_type_with_dataframe_argument "
            + "(tests.integration.test_integration_frameit."
            + "TestIntegrationForDataFrameArgument)"
        )

        # ACT
        self.dummy(self.pldf, multiplier=2)

        cache = inspect_cache().pop(0)
        args = cache.get("args")

        # ASSERT
        self.assertEqual(cache["mod"], "tests.integration.test_integration_frameit")
        self.assertEqual(cache["fun"], "dummy")
        self.assertEqual(cache["pkg"], "polars")

        self.assertEqual(args["self"], argself)
        self.assertEqual(args["df"], str(self.pldf))
        self.assertEqual(args["multiplier"], "2")

    def test_full_integration_for_pandas_return_type_with_dataframe_argument(self):
        # ARRANGE
        argself = (
            "test_full_integration_for_pandas_return_type_with_dataframe_argument "
            + "(tests.integration.test_integration_frameit."
            + "TestIntegrationForDataFrameArgument)"
        )

        # ACT
        self.dummy(self.pddf, multiplier=2)

        cache: dict = inspect_cache().pop(0)
        args = cache.get("args")

        # ASSERT
        self.assertEqual(cache["mod"], "tests.integration.test_integration_frameit")
        self.assertEqual(cache["fun"], "dummy")
        self.assertEqual(cache["pkg"], "pandas")

        self.assertEqual(args["self"], argself)
        self.assertEqual(args["df"], str(self.pddf))
        self.assertEqual(args["multiplier"], "2")

    @patch("polars.read_parquet", wraps=pl.read_parquet)
    def test_full_integration_for_iterable_polars_return_type_with_dataframe_argument(
        self, mock_polars_read_parquet
    ):
        # ARRANGE
        argself = (
            "test_full_integration_for_iterable_polars_return_type_with_dataframe_argument "
            + "(tests.integration.test_integration_frameit."
            + "TestIntegrationForDataFrameArgument)"
        )

        # ACT
        results = self.iterable_dummy(self.pldf, multipliers=[1, 2, 3, 4])
        expectations = self.iterable_dummy(self.pldf, multipliers=[1, 2, 3, 4])

        cache: dict = inspect_cache().pop(0)
        args = cache.get("args")

        # ASSERT
        for result, expectation in zip(results, expectations):
            result.frame_equal(expectation)

        mock_polars_read_parquet.assert_any_call(f"{PATH}/{cache['files'][0]}")
        mock_polars_read_parquet.assert_any_call(f"{PATH}/{cache['files'][1]}")
        mock_polars_read_parquet.assert_any_call(f"{PATH}/{cache['files'][2]}")
        mock_polars_read_parquet.assert_any_call(f"{PATH}/{cache['files'][3]}")

        self.assertEqual(cache["mod"], "tests.integration.test_integration_frameit")
        self.assertEqual(cache["fun"], "iterable_dummy")
        self.assertEqual(cache["pkg"], "polars")

        self.assertEqual(args["self"], argself)
        self.assertEqual(args["df"], str(self.pldf))
        self.assertEqual(args["multipliers"], "[1, 2, 3, 4]")

    @patch("pandas.read_parquet", wraps=pd.read_parquet)
    def test_full_integration_for_iterable_pandas_return_type_with_dataframe_argument(
        self, mock_pandas_read_parquet
    ):
        # ARRANGE
        argself = (
            "test_full_integration_for_iterable_pandas_return_type_with_dataframe_argument "
            + "(tests.integration.test_integration_frameit."
            + "TestIntegrationForDataFrameArgument)"
        )

        # ACT
        results = self.iterable_dummy(self.pddf, multipliers=[1, 2, 3, 4])
        expectations = self.iterable_dummy(self.pddf, multipliers=[1, 2, 3, 4])

        cache = inspect_cache().pop(0)
        args = cache.get("args")

        # ASSERT
        for result, expectation in zip(results, expectations):
            pd.testing.assert_frame_equal(result, expectation)

        mock_pandas_read_parquet.assert_any_call(f"{PATH}/{cache['files'][0]}")
        mock_pandas_read_parquet.assert_any_call(f"{PATH}/{cache['files'][1]}")
        mock_pandas_read_parquet.assert_any_call(f"{PATH}/{cache['files'][2]}")
        mock_pandas_read_parquet.assert_any_call(f"{PATH}/{cache['files'][3]}")

        self.assertEqual(cache["mod"], "tests.integration.test_integration_frameit")
        self.assertEqual(cache["fun"], "iterable_dummy")
        self.assertEqual(cache["pkg"], "pandas")

        self.assertEqual(args["self"], argself)
        self.assertEqual(args["df"], str(self.pddf))
        self.assertEqual(args["multipliers"], "[1, 2, 3, 4]")

    @patch("polars.read_parquet", wraps=pl.read_parquet)
    def test_similar_dataframe_not_registered_as_cached_for_polars_dataframe_arguments(
        self, mock_polars_read_parquet
    ):
        # ARRANGE (& ACT)
        frameA = self.dummy(self.pldf, multiplier=2)
        frameB = self.dummy(self.pldf + 1, multiplier=2)

        cache = inspect_cache()

        # ASSERT
        mock_polars_read_parquet.assert_not_called()

        # Verify that we have two entries in our file cache
        self.assertEqual(len(cache), 2)
        self.assertFalse(frameA.frame_equal(frameB))
