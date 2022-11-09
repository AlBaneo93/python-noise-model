import train
import train_vib
import traceback

if __name__ == '__main__':
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
