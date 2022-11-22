import time


def Duration(func):
    def wrap(*args):
        func_name: str = func.__name__
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        # print(f"{func_name} -- {end - start:.4f} sec elapsed")
        return result

    return wrap