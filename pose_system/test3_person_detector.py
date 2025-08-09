# test_person_detector.py

import cv2
from input_handler import InputHandler
from person_detector import PersonDetector

def main():
    # 0은 기본 웹캠
    handler = InputHandler(source=0)
    detector = PersonDetector()

    print("Person Detector 테스트: 'q' 를 눌러 종료하세요.")
    while True:
        frame = handler.get_frame()
        if frame is None:
            print("프레임 읽기 실패, 종료합니다.")
            break

        # 사람 검출
        boxes = detector.detect(frame)

        # 바운딩 박스 그리기
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Person Detector Test", frame)

        # 1ms 대기, 'q' 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("테스트 종료")
            break

    handler.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
