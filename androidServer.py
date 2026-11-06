from datetime import datetime
from collections import Counter
import time
import threading
from flask import Flask, Response, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("results/best.pt")
cap = cv2.VideoCapture(0)

# Глобальные переменные для телеметрии
latest_telemetry = {
    "fps": 0.0,
    "inference_time_ms": 0.0,
    "total_detected": 0,
    "class_counts": {},
    "purity_score": 100.0,
    "status": "Ожидание потока..."
}
latest_logs = []
lock = threading.Lock()


def process_frames():
    global latest_telemetry, latest_logs
    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        start_time = time.time()

        # Инференс модели
        results = model(frame, conf=0.68)

        inference_time = (time.time() - start_time) * 1000
        current_fps = 1.0 / (time.time() - prev_time) if (time.time() - prev_time) > 0 else 0.0
        prev_time = time.time()

        box_data = results[0].boxes
        class_names = model.names
        detected_items = []
        class_counts = {}

        if box_data is not None and len(box_data) > 0:
            for box in box_data:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf >= 0.7:
                    name = class_names.get(cls_id, str(cls_id))
                    detected_items.append(name)

        if detected_items:
            counts = Counter(detected_items)
            class_counts = dict(counts)
            total = len(detected_items)

            # Пример расчета чистоты (предполагаем, что дефекты содержат слова вроде "broken", "weed", "damaged" или берем общую массу)
            # Здесь можно настроить логику под названия ваших классов в model.names
            defective_count = sum(count for name, count in class_counts.items() if
                                  any(bad in name.lower() for bad in ['broken', 'weed', 'damage', 'сор', 'битое']))
            purity = max(0.0, ((total - defective_count) / total) * 100) if total > 0 else 100.0

            timestamp = datetime.now().strftime("%H:%M:%S")
            log_msg = f"[{timestamp}] Партия: всего {total} шт. | Чистота: {purity:.1f}%"

            with lock:
                latest_telemetry = {
                    "fps": round(current_fps, 1),
                    "inference_time_ms": round(inference_time, 1),
                    "total_detected": total,
                    "class_counts": class_counts,
                    "purity_score": round(purity, 1),
                    "status": "АНАЛИЗ АКТИВЕН"
                }
                latest_logs.append(log_msg)
                if len(latest_logs) > 30:
                    latest_logs.pop(0)
        else:
            with lock:
                latest_telemetry["fps"] = round(current_fps, 1)
                latest_telemetry["inference_time_ms"] = round(inference_time, 1)
                latest_telemetry["status"] = "ПОТОК ЧИСТ / ЗЕРНО НЕ ОБНАРУЖЕНО"


# Запуск обработки в фоне
threading.Thread(target=process_frames, daemon=True).start()


@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>AGRO-VISION // Бортовой терминал комбайна</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body {
                background-color: #121c14; /* Темно-зеленый агро-фон */
                color: #e2e8f0;
                font-family: 'Roboto Condensed', sans-serif;
                margin: 0;
                padding: 15px;
            }
            header {
                background-color: #1e3a1e;
                border-bottom: 4px solid #4ade80;
                padding: 15px 25px;
                border-radius: 6px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            }
            header h1 {
                margin: 0;
                color: #facc15; /* Фермерский желтый */
                font-size: 1.6rem;
                letter-spacing: 1px;
            }
            .status-badge {
                background: #166534;
                color: #4ade80;
                padding: 6px 14px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 1.1rem;
                border: 1px solid #4ade80;
            }
            .main-grid {
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 20px;
                margin-top: 20px;
            }
            .card {
                background: #192b1a;
                border: 2px solid #2d4a2e;
                border-radius: 6px;
                padding: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            }
            .card-title {
                font-size: 1.2rem;
                color: #facc15;
                margin-bottom: 15px;
                border-bottom: 1px solid #2d4a2e;
                padding-bottom: 8px;
                text-transform: uppercase;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
            }
            .metric-box {
                background: #122013;
                border: 1px solid #365337;
                padding: 15px;
                border-radius: 4px;
                text-align: center;
            }
            .metric-value {
                font-size: 2rem;
                font-weight: bold;
                color: #4ade80;
                margin-top: 5px;
            }
            .metric-value.warning {
                color: #f87171;
            }
            .classes-list {
                margin-top: 15px;
                max-height: 200px;
                overflow-y: auto;
            }
            .class-item {
                display: flex;
                justify-content: space-between;
                padding: 8px 10px;
                background: #122013;
                margin-bottom: 6px;
                border-radius: 4px;
                border-left: 4px solid #4ade80;
            }
            .log-box {
                background: #0d170e;
                height: 380px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 0.95rem;
                padding: 12px;
                border-radius: 4px;
                border: 1px solid #2d4a2e;
                color: #a3e635;
            }
            .log-entry {
                margin-bottom: 6px;
                border-bottom: 1px dotted #1e3a1e;
                padding-bottom: 3px;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>ZERNO // БОРТОВОЙ ТЕРМИНАЛ</h1>
            <div class="status-badge" id="statusBadge">СИСТЕМА АКТИВНА</div>
        </header>

        <div class="main-grid">
            <!-- Левая колонка: Основные метрики зерна -->
            <div class="card">
                <div class="card-title">Оперативные показатели потока</div>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div>Чистота партии</div>
                        <div class="metric-value" id="purityVal">100.0%</div>
                    </div>
                    <div class="metric-box">
                        <div>Скорость обработки</div>
                        <div class="metric-value" id="fpsVal">0 FPS</div>
                    </div>
                    <div class="metric-box">
                        <div>Время инференса (кадр)</div>
                        <div class="metric-value" id="timeVal">0 ms</div>
                    </div>
                    <div class="metric-box">
                        <div>Объектов в кадре</div>
                        <div class="metric-value" id="totalVal">0</div>
                    </div>
                </div>

                <div class="card-title" style="margin-top: 25px;">Детализация классов (детекция YOLO)</div>
                <div class="classes-list" id="classesContainer">
                    <div style="color: #6b7280; text-align: center; padding: 20px;">Ожидание данных с камеры...</div>
                </div>
            </div>

            <!-- Правая колонка: Журнал сессии -->
            <div class="card">
                <div class="card-title">Журнал телеметрии</div>
                <div class="log-box" id="logContainer">
                    <div class="log-entry">[СИСТЕМА] Инициализация модели YOLO на борту...</div>
                </div>
            </div>
        </div>

        <script>
            setInterval(async () => {
                try {
                    const res = await fetch('/api/telemetry');
                    const data = await res.json();

                    // Обновляем метрики
                    document.getElementById('purityVal').innerText = data.purity_score + '%';
                    document.getElementById('fpsVal').innerText = data.fps + ' FPS';
                    document.getElementById('timeVal').innerText = data.inference_time_ms + ' ms';
                    document.getElementById('totalVal').innerText = data.total_detected;
                    document.getElementById('statusBadge').innerText = data.status;

                    // Окраска чистоты тревожным цветом, если падает ниже 90%
                    const purityEl = document.getElementById('purityVal');
                    if (data.purity_score < 90) {
                        purityEl.classList.add('warning');
                    } else {
                        purityEl.classList.remove('warning');
                    }

                    // Список классов
                    const classContainer = document.getElementById('classesContainer');
                    const counts = data.class_counts;
                    if (Object.keys(counts).length > 0) {
                        classContainer.innerHTML = Object.entries(counts)
                            .map(([name, count]) => `<div class="class-item"><span>${name}</span><strong>${count} шт.</strong></div>`)
                            .join('');
                    } else {
                        classContainer.innerHTML = '<div style="color: #6b7280; text-align: center; padding: 15px;">Зерно в поле зрения не зафиксировано</div>';
                    }

                    // Логи
                    const logContainer = document.getElementById('logContainer');
                    if (data.logs && data.logs.length > 0) {
                        logContainer.innerHTML = data.logs.map(log => `<div class="log-entry">${log}</div>`).join('');
                        logContainer.scrollTop = logContainer.scrollHeight;
                    }
                } catch (e) {
                    console.error("Ошибка опроса API:", e);
                }
            }, 1000);
        </script>
    </body>
    </html>
    '''


@app.route('/api/telemetry')
def api_telemetry():
    with lock:
        data = latest_telemetry.copy()
        data["logs"] = latest_logs
        return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)