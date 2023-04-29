import cv2
import pytesseract
from pytesseract import Output
import numpy as np

def color_filter(frame, color):
    if color == "yellow":
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
    elif color == "green":
        lower = np.array([40, 100, 100])
        upper = np.array([70, 255, 255])
    elif color == "blue":
        lower = np.array([100, 150, 0])
        upper = np.array([140, 255, 255])
    else:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper) # Эта строка создает бинарную маску, в которой белые пиксели (значение 255) соответствуют пикселям исходного изображения hsv, у которых значения находятся в диапазоне между lower и upper. lower и upper - это массивы, содержащие нижние и верхние границы диапазона для каждого из трех каналов HSV. В данном случае, lower и upper представляют диапазоны желтого, зеленого или синего цвета, в зависимости от текущей итерации цикла.
    return mask
#В результате, маска mask будет содержать только пиксели, соответствующие определенному диапазону цвета. Эта маска затем используется для выделения интересующих нас объектов на исходном изображении.
def recognize_digits(frame):
    config = "--psm 4 -c tessedit_char_whitelist=012"
    result = pytesseract.image_to_data(frame, config=config, output_type=Output.DICT)
    return result

def main():
    cap = cv2.VideoCapture(0)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    x, y, w, h = 100, 100, 600, 600

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi = frame[y:y+h, x:x+w]

        for color in ["yellow", "green", "blue"]:
            mask = color_filter(roi, color)
            masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
            gray = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            digits_data = recognize_digits(thresh)

            for i in range(len(digits_data["text"])):
                if digits_data["conf"][i] > 80:
                    x_digit, y_digit, w_digit, h_digit = digits_data["left"][i], digits_data["top"][i], digits_data["width"][i], digits_data["height"][i]
                    cv2.rectangle(roi, (x_digit, y_digit), (x_digit + w_digit, y_digit + h_digit), (0, 255, 0), 2)
                    cv2.putText(frame, f"{digits_data['text'][i]} {color}", (x + x_digit, y + y_digit - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        cv2.imshow("ROI", roi)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('w'):
            y = max(y - 5, 0)
        elif key == ord('s'):
            y = min(y + 5, frame_height - h)
        elif key == ord('a'):
            x = max(x - 5, 0)
        elif key == ord('d'):
            x = min(x + 5, frame_width - w)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


