# test_input.py

import cv2
from input_handler import InputHandler

def main():
    # 0은 기본 웹캠, 파일 경로를 넣으면 동영상 파일도 테스트 가능
    handler = InputHandler(source=0)

    print("웹캠 화면이 나타나면 'q'를 눌러 종료하세요.")
    while True:
        frame = handler.get_frame()
        if frame is None:
            print("프레임을 읽어올 수 없습니다. 종료합니다.")
            break

        # 원본 프레임 출력
        cv2.imshow("InputHandler Test", frame)

        # 1ms 대기, 'q' 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("사용자 종료 요청 - 테스트 종료")
            break

    # 리소스 해제
    handler.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
