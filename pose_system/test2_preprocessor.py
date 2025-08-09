# test_preprocessor.py

import cv2
from input_handler import InputHandler
from preprocessor import Preprocessor

def main():
    # 0은 기본 웹캠
    handler = InputHandler(source=0)
    preproc = Preprocessor()  # 기본 ROI 사용

    print("웹캠의 원본과 전처리된 영상을 확인하세요. 'q' 눌러 종료.")

    while True:
        frame = handler.get_frame()
        if frame is None:
            print("프레임을 읽을 수 없습니다.")
            break

        # 전처리 실행
        processed_rgb = preproc.preprocess(frame)

        # 원본과 전처리된 영역(ROI)을 함께 시각화
        cv2.imshow("Original", cv2.resize(frame, (640, 480)))

        # RGB -> BGR로 변환하여 올바르게 표시
        display_img = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("Preprocessed (ROI)", display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("테스트 종료")
            break

    handler.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
