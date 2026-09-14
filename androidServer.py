from datetime import datetime
from collections import Counter
import threading
from flask import Flask, Response, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("results/grain_finetune-2/weights/best.pt")
cap = cv2.VideoCapture(0)

current_frame = None
latest_logs = []
lock = threading.Lock()


def capture_frames():
    global current_frame, latest_logs
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Инференс с фильтром уверенности от 70% (conf=0.7)
        results = model(frame, conf=0.68)
        annotated_frame = results[0].plot()

        # Сбор информации о детекциях для логов
        box_data = results[0].boxes
        class_names = model.names
        detected_items = []

        if box_data is not None and len(box_data) > 0:
            for box in box_data:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf >= 0.7:
                    name = class_names.get(cls_id, str(cls_id))
                    detected_items.append(name)

        # Если найдены объекты с высокой уверенностью, формируем красивый лог
        if detected_items:
            counts = Counter(detected_items)
            counts_str = ", ".join([f"{item}: {count}" for item, count in counts.items()])
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_msg = f"[{timestamp}] Детекция -> {counts_str}"

            with lock:
                latest_logs.append(log_msg)
                if len(latest_logs) > 35:  # Ограничение истории логов
                    latest_logs.pop(0)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        with lock:
            current_frame = buffer.tobytes()


# Запуск захвата видео в фоновом потоке
threading.Thread(target=capture_frames, daemon=True).start()


@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Command Center // AI Vision Matrix</title>
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@500;700&display=swap" rel="stylesheet">
        <style>
            body {
                background-color: #0b0e14;
                color: #c9d1d9;
                font-family: 'Rajdhani', sans-serif;
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            h1 {
                font-family: 'Orbitron', sans-serif;
                color: #00ffcc;
                text-transform: uppercase;
                letter-spacing: 2px;
                margin-bottom: 20px;
                text-shadow: 0 0 10px rgba(0,255,204,0.3);
                font-size: 1.8rem;
            }
            .container {
                display: flex;
                gap: 20px;
                width: 100%;
                max-width: 1350px;
            }
            .video-box {
                flex: 2;
                background: #121820;
                border: 2px solid #1f2937;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 0 25px rgba(0,0,0,0.6);
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .video-box img {
                max-width: 100%;
                border-radius: 4px;
                border: 1px solid #374151;
            }
            .log-box {
                flex: 1;
                background: #121820;
                border: 2px solid #1f2937;
                border-radius: 8px;
                padding: 15px;
                display: flex;
                flex-direction: column;
                box-shadow: 0 0 25px rgba(0,0,0,0.6);
            }
            .log-title {
                font-family: 'Orbitron', sans-serif;
                font-size: 1.1rem;
                color: #ec4899;
                margin-bottom: 10px;
                border-bottom: 1px solid #1f2937;
                padding-bottom: 8px;
                text-shadow: 0 0 8px rgba(236,72,153,0.3);
            }
            .log-content {
                flex-grow: 1;
                height: 440px;
                overflow-y: auto;
                font-family: 'Courier New', Courier, monospace;
                font-size: 0.9rem;
                background: #070a0f;
                padding: 10px;
                border-radius: 4px;
                border: 1px solid #111827;
                color: #38bdf8;
            }
            .log-entry {
                margin-bottom: 6px;
                border-bottom: 1px dashed #1f2937;
                padding-bottom: 4px;
            }
        </style>
    </head>
    <body>
        <h1>Command Center // Neural Vision</h1>
        <div class="container">
            <div class="video-box">
                <img src="/video_feed" alt="Live Feed">
            </div>
            <div class="log-box">
                <div class="log-title">Telemetry & Detection Log</div>
                <div class="log-content" id="logContainer">
                    <div class="log-entry">[SYSTEM] Initializing AI matrix (Threshold: 68%)...</div>
                </div>
            </div>
        </div>
        <script>
            // Автоматическое обновление логов каждую секунду
            setInterval(async () => {
                try {
                    const res = await fetch('/api/logs');
                    const data = await res.json();
                    const container = document.getElementById('logContainer');
                    if (data.logs && data.logs.length > 0) {
                        container.innerHTML = data.logs.map(log => `<div class="log-entry">${log}</div>`).join('');
                        container.scrollTop = container.scrollHeight;
                    }
                } catch (e) {
                    console.error("Log fetch error:", e);
                }
            }, 1000);
        </script>
    </body>
    </html>
    '''


@app.route('/api/logs')
def api_logs():
    with lock:
        return jsonify(logs=latest_logs)


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