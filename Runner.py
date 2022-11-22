import os.path
import traceback

import train
import train_vib
from Constants import noise_train_prefix, vibe_train_prefix


def check_need_dirs(dir_path: str) -> bool:
    exists = os.path.exists(dir_path) and os.path.isdir(dir_path)
    if not exists:
        os.mkdir(dir_path)
    return exists


def whole_tasks():
    if not check_need_dirs(noise_train_prefix):
        print(f"{noise_train_prefix} 경로가 존재 하지 않습니다.")
        return

    if not check_need_dirs(vibe_train_prefix):
        print(f"{vibe_train_prefix} 경로가 존재 하지 않습니다.")
        return

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
