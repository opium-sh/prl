from functools import wraps
from time import perf_counter

from prl.utils.loggers import time_logger


def timeit(func, profiled_function_name=None):
    """
    Decorator for profiling execution time for the functions and methods. To measure time of a method or function you
    have to put @timeit in line nefore function, or wrap a function in the code:

    @timeit
    def func(a, b, c="1"):
        pass

    or in the code:

    result = timeit(func, profiled_function_name="Profiled function func")(5,5)

    To print results of measurment you have to print time_logger object from this package at the end
    of the program execution. When the name of the function can be ambiguous in the profiler
    data use profiled_function_name parameter.

    Args:
        func: function, which execution time we wan to measure
        profiled_function_name: user defined name for the wrapped function.

    Returns:
        wrapped function

    """

    @wraps(func)
    def timed(*args, **kwargs):
        nonlocal profiled_function_name
        start_time = perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = perf_counter() - start_time
        if profiled_function_name is None:
            profiled_function_name = func.__qualname__
        time_logger.add(profiled_function_name, time_elapsed)
        return result

    return timed
