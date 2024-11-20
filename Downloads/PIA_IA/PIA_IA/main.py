import json
import cv2
from ultralytics import YOLOv10
import numpy as np
import math
import re
import os
from datetime import datetime
from paddleocr import PaddleOCR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if not os.path.exists('json'):
    os.makedirs('json')

cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()


model = YOLOv10("C:\\Users\\admin\\Downloads\\PIA_IA\\PIA_IA\\weights\\best.pt")
count = 0
classNames = ["License"]
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

letter_to_number_map = {
    'O': '0',
    'A': '4',
    'I': '1',
    'L': '1',
    'l': '1',
    'E': '3',
    'S': '5',
    's': '5',
    'G': '6',
    'B': '8',
    'Z': '2',
    'z': '2',
    'T': '7',
    'C': '6',
    'q': '9',
    'D': '0',
    'X': '8'
}

def paddle_ocr(frame, x1, y1, x2, y2):
    frame = frame[y1:y2, x1:x2]
    
    result = ocr.ocr(frame, det=False, rec=True, cls=False)
    
    text = ""
    for r in result:
        scores = r[0][1]
        if np.isnan(scores):
            scores = 0
        else:
            scores = int(scores * 100)
        
        if scores > 60:
            text = r[0][0]
    
    text = ''.join([letter_to_number_map.get(char.upper(), char) for char in text])
    
    pattern = re.compile('[\W]')
    text = pattern.sub('', text)
    
    text = text.replace("???", "")  # Eliminar secuencias de "???"
    text = text.replace("粤", "")  # Eliminar caracteres no deseados

    return str(text)

def save_json(license_plates, startTime, endTime):
    interval_data = {
        "Start Time": startTime.isoformat(),
        "End Time": endTime.isoformat(),
        "License Plate": list(license_plates)
    }
    interval_file_path = f"json/output_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(interval_file_path, 'w') as f:
        json.dump(interval_data, f, indent=2)

    # Guardar datos acumulativos en archivo JSON
    cummulative_file_path = "json/LicensePlateData.json"
    if os.path.exists(cummulative_file_path):
        with open(cummulative_file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(interval_data)

    with open(cummulative_file_path, 'w') as f:
        json.dump(existing_data, f, indent=2)





# Variables para la gestión del tiempo
startTime = datetime.now()
license_plates = set()

# Bucle principal de procesamiento de video
while True:
    ret, frame = cap.read()
    if ret:
        currentTime = datetime.now()
        count += 1
        print(f"Frame Number: {count}")
        results = model.predict(frame, conf=0.45)  
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Coordenadas de la caja delimitadora
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                classNameInt = int(box.cls[0])
                clsName = classNames[classNameInt]
                conf = math.ceil(box.conf[0] * 100) / 100
                # Obtener la matrícula mediante OCR
                label = paddle_ocr(frame, x1, y1, x2, y2)
                if label:
                    license_plates.add(label)
                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        
        # Cada 5 segundos guarda los datos
        if (currentTime - startTime).seconds >= 5:
            endTime = currentTime
            save_json(license_plates, startTime, endTime)
            startTime = currentTime
            license_plates.clear()

        cv2.imshow("Camera Feed", frame)
        #LOS FRAMES SON CADA SEGUNDO, NO LO MUEVAN PQ EXPLOTA
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
