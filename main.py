from ultralytics import YOLO
import cv2

# Загружаем вашу обученную модель
model = YOLO(r"C:\Users\User\runs\detect\train-11\weights\best.pt")

image_path = r"humanCheck/kovarnyy-fuzarioz-kolosa_1.jpg"

# уверенность модели в 70 процентов+
results = model(image_path, conf=0.7, show=False, save=True)

for r in results:
    print(f"Найдено объектов с уверенностью 70%+: {len(r.boxes)}")

    annotated_frame = r.plot()

    window_name = "Grain Detection Result (Conf >= 0.7)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)

    cv2.imshow(window_name, annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()