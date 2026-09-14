import os
import time
from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO

# --- НАСТРОЙКИ ПУТЕЙ ---
MODEL_PATH = r"C:\Users\User\Desktop\zerno\results\grain_finetune-2\weights\best.pt"
IMAGES_DIR = (
    r"C:\Users\User\Desktop\zerno\testApp\testImgs\train\images"  # или train
)
LABELS_DIR = r"C:\Users\User\Desktop\zerno\testApp\testImgs\train\labels"
OUTPUT_PRED_DIR = r"C:\Users\User\Desktop\zerno\testApp\predicted_images"
OUTPUT_ERROR_DIR = r"C:\Users\User\Desktop\zerno\testApp\error_frames"
OUTPUT_EXCEL = r"C:\Users\User\Desktop\zerno\testApp\zerno_full_validation_report.xlsx"

os.makedirs(OUTPUT_PRED_DIR, exist_ok=True)
os.makedirs(OUTPUT_ERROR_DIR, exist_ok=True)

# Загружаем модель
print("Загрузка модели...")
model = YOLO(MODEL_PATH)

# Строгая привязка классов согласно вашему уточнению:
# 0 - Пшеница, 1 - Ячмень, 2 - Рожь
class_names = {0: "Пшеница (кл. 0)", 1: "Ячмень (кл. 1)", 2: "Рожь (кл. 2)"}

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".heic")
image_path_obj = Path(IMAGES_DIR)

if not image_path_obj.exists():
  print(f"ОШИБКА: Папка с изображениями не найдена: {IMAGES_DIR}")
  exit(1)

image_paths = [
    p for p in image_path_obj.glob("*") if p.suffix.lower() in valid_extensions
]
print(f"Найдено изображений для тестирования: {len(image_paths)}")

if len(image_paths) == 0:
  print("Нет изображений для теста!")
  exit(0)

results_data = []

# Счетчики для итогов
total_time = 0
total_gt_count = 0
total_pred_count = 0
all_confidences = []
total_missed = 0
total_false_alarms = 0

# Глобальные счетчики по каждому классу для итоговой строки
global_gt_by_class = {c_name: 0 for c_name in class_names.values()}
global_pred_by_class = {c_name: 0 for c_name in class_names.values()}

for img_path in image_paths:
  image_name = img_path.name
  txt_name = img_path.stem + ".txt"
  txt_path = Path(LABELS_DIR) / txt_name

  img = cv2.imread(str(img_path))
  if img is None:
    continue

  # Читаем реальную разметку (Ground Truth) из .txt
  gt_classes = []
  if txt_path.exists():
    for line in txt_path.read_text(encoding="utf-8").splitlines():
      if not line.strip():
        continue
      parts = line.strip().split()
      cls_id = int(parts[0])
      gt_classes.append(cls_id)

  # Замер времени инференса (порог 61%)
  start_time = time.time()
  results = model(img, conf=0.61, verbose=False)
  processing_time_ms = (time.time() - start_time) * 1000
  total_time += processing_time_ms

  r = results[0]
  pred_boxes = r.boxes

  pred_classes = []
  confidences = []

  for box in pred_boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    pred_classes.append(cls_id)
    confidences.append(conf)
    all_confidences.append(conf)

  # Сохраняем визуализацию (размеченную картинку с рамками)
  annotated_img = r.plot()
  save_pred_path = os.path.join(OUTPUT_PRED_DIR, image_name)
  if save_pred_path.lower().endswith(".heic"):
    save_pred_path = str(Path(save_pred_path).with_suffix(".jpg"))
  cv2.imwrite(save_pred_path, annotated_img)

  # Считаем количество по каждому классу для кадра
  frame_gt = {c_name: 0 for c_name in class_names.values()}
  for cid in gt_classes:
    c_name = class_names.get(cid, f"Неизвестный ({cid})")
    if c_name in frame_gt:
      frame_gt[c_name] += 1
      global_gt_by_class[c_name] += 1

  frame_pred = {c_name: 0 for c_name in class_names.values()}
  for cid in pred_classes:
    c_name = class_names.get(cid, f"Неизвестный ({cid})")
    if c_name in frame_pred:
      frame_pred[c_name] += 1
      global_pred_by_class[c_name] += 1

  gt_count = len(gt_classes)
  pred_count = len(pred_classes)
  total_gt_count += gt_count
  total_pred_count += pred_count

  missed_objects = max(0, gt_count - pred_count)
  false_alarms = max(0, pred_count - gt_count)

  total_missed += missed_objects
  total_false_alarms += false_alarms

  avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

  is_problematic = "Нет"
  if missed_objects > 0 or false_alarms > 0 or avg_conf < 0.70:
    is_problematic = "Да"
    error_save_path = os.path.join(OUTPUT_ERROR_DIR, image_name)
    if error_save_path.lower().endswith(".heic"):
      error_save_path = str(Path(error_save_path).with_suffix(".jpg"))
    cv2.imwrite(error_save_path, img)

  # Формируем строку таблицы
  row = {
      "Имя файла": image_name,
      "Время обработки (мс)": round(processing_time_ms, 2),
      "Всего в разметке (GT)": gt_count,
      "Всего найдено (Pred)": pred_count,
  }

  # Добавляем колонки для видов зерна (Пшеница, Ячмень, Рожь)
  for c_name in class_names.values():
    g_val = frame_gt[c_name]
    p_val = frame_pred[c_name]
    row[f"Вид: {c_name} (Найдено / План)"] = f"{p_val} / {g_val}"

  row.update({
      "Пропущено (False Neg)": missed_objects,
      "Ложные срабатывания (False Pos)": false_alarms,
      "Средний Confidence": round(avg_conf, 3),
      "Проблемный кадр": is_problematic,
  })

  results_data.append(row)

# Создаем DataFrame
df = pd.DataFrame(results_data)

# --- ИТОГОВАЯ СТРОКА ---
avg_time_val = round(total_time / len(image_paths), 2)
overall_avg_conf = round(
    sum(all_confidences) / len(all_confidences) if all_confidences else 0.0, 3
)
detection_rate = (
    round((total_pred_count / total_gt_count) * 100, 1)
    if total_gt_count > 0
    else 0
)

summary_row = {
    "Имя файла": "ИТОГО / СРЕДНЕЕ ПО НАБОРУ",
    "Время обработки (мс)": avg_time_val,
    "Всего в разметке (GT)": total_gt_count,
    "Всего найдено (Pred)": total_pred_count,
}

for c_name in class_names.values():
  g_sum = global_gt_by_class[c_name]
  p_sum = global_pred_by_class[c_name]
  summary_row[f"Вид: {c_name} (Найдено / План)"] = f"{p_sum} / {g_sum}"

summary_row.update({
    "Пропущено (False Neg)": total_missed,
    "Ложные срабатывания (False Pos)": total_false_alarms,
    "Средний Confidence": overall_avg_conf,
    "Проблемный кадр": f"Покрытие: {detection_rate}%",
})

summary_df = pd.DataFrame([summary_row])
df = pd.concat([df, summary_df], ignore_index=True)

# Экспорт в Excel
df.to_excel(OUTPUT_EXCEL, index=False)

print(f"\nТестирование и валидация успешно завершены!")
print(f"Отчет сохранен в файл: {OUTPUT_EXCEL}")
print(f"Размеченные картинки сохранены в: {OUTPUT_PRED_DIR}/")
print(f"Склонность модели к обнаружению: {detection_rate}% от реального объема")