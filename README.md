# frameit

Frameit is a DataFrame caching tool that is designed to work on pandas and polars data objects. The tool is intended to be simple
and make the process of running expensive functions that are called several times in a program less painful.



## Examples
```py
import pandas as pd

from frameit import frameit

@frameit
def some_expensive_function():
    df = pd.DataFrame()
    ...
    return df

```

```py
import polars as pl

from frameit import frameit

@frameit(max_size=5e6, max_time=20)
def some_expensive_function():
    df1 = pl.DataFrame()
    df2 = pl.DataFrame()
    df3 = pl.DataFrame()
    ...
    return df1, df2, df3

```

```py
import pandas as pd

from frameit import frameit, clear_cache


if __name__ == '__main__':


    while True:
        # Some code that calls a set of expensive functions
        # that produce some dataframes that are relatively
        # static
        ...

    # End of the program
    clear_cache()
```
