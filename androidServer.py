import threading
from flask import Flask, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("results/grain_finetune-2/weights/best.pt")
cap = cv2.VideoCapture(0)

current_frame = None
lock = threading.Lock()

def capture_frames():
    global current_frame
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        with lock:
            current_frame = buffer.tobytes()

# Запускаем захват видео в отдельном потоке, чтобы не тормозить сервер
threading.Thread(target=capture_frames, daemon=True).start()

@app.route('/')
def index():
    return "<html><body><center><h1>Zerno Detection</h1><img src='/video_feed'></center></body></html>"

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if current_frame is None:
                    continue
                frame = current_frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/frame')
def frame():
    with lock:
        if current_frame is None:
            return "", 204
        return Response(current_frame, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)