import sys
import time

import elapsed as elapsed

import Logger
import train
import train_vib
import traceback


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


@Duration
def whole_tasks():
    try:
        train.main()
    except Exception as e:
        print(f"[NOISE] 오류 발생")
        traceback.print_exc()
    try:
        train_vib.main()
    except Exception as e:
        print(f"[VIBE] 오류 발생")
        traceback.print_exc()


if __name__ == '__main__':
    whole_tasks()
