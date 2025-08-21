import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 임시! 장기적으로 비권장
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'main'], required=True, help="실행 모드 선택")
    args = parser.parse_args()

    if args.mode == 'train':
        print("🚀 Training mode 실행")
        # train 함수 호출
    elif args.mode == 'main':
        from gui.interface import main_interface
        main_interface()

if __name__ == '__main__':

    main()
