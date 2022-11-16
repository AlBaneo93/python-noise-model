import time


def Duration(func):
    def wrap(*args):
        from Logger import logger
        func_name: str = func.__name__
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        logger.info(f"{func_name} -- {end - start:.4f} sec elapsed")
        return result

    return wrap