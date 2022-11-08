import train
import train_vib

if __name__ == '__main__':
    try:
        train.main()
    except:
        print(f"[NOISE] 오류 발생")
    try:
        train_vib.main()
    except:
        print(f"[VIBE] 오류 발생")
