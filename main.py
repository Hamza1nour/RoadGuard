import cv2
import time
import math
import csv
from ultralytics import YOLO
from tracker import Tracker


model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture("Car_Vehicles.mp4")
if not cap.isOpened():
    print("❌ فشل في تحميل الفيديو. تأكد من وجود الملف.")
    exit()


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (800, 400))


tracker = Tracker()


csv_file = open("violations.csv", "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Vehicle_ID", "Speed", "Time", "Direction"])

fps = cap.get(cv2.CAP_PROP_FPS)
pixel_to_meter = 0.08
frame_count = 0
overspeed_count = 0
overspeed_vehicles = set()

vehicle_speeds = {}
vehicle_last_position = {}
vehicle_last_time = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (800, 400))
    frame_height, frame_width = frame.shape[:2]

    if frame_count % 3 == 0:
        results = model.predict(frame, verbose=False)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append([x1, y1, x2 - x1, y2 - y1])

        tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            x, y, w, h, vehicle_id = obj
            cx = x + w // 2
            cy = y + h // 2

            if vehicle_id in vehicle_last_position:
                prev_x, prev_y = vehicle_last_position[vehicle_id]
                prev_time = vehicle_last_time[vehicle_id]
                distance = math.hypot(cx - prev_x, cy - prev_y) * pixel_to_meter
                elapsed_time = (frame_count - prev_time) / fps
                speed = distance / elapsed_time * 3.6 if elapsed_time > 0 else 0
                direction = "Up" if cy < prev_y else "Down"
            else:
                speed = 0
                direction = "Unknown"

            vehicle_last_position[vehicle_id] = (cx, cy)
            vehicle_last_time[vehicle_id] = frame_count
            vehicle_speeds[vehicle_id] = speed

            
            if cx > frame_width * 0.6 and direction == "Up":
                continue

            
            if cy > frame_height * 0.6 and speed > 60:
                if vehicle_id not in overspeed_vehicles:
                    overspeed_count += 1
                    overspeed_vehicles.add(vehicle_id)
                    csv_writer.writerow([vehicle_id, int(speed), time.strftime("%H:%M:%S"), direction])
                info_text = f"ID: {vehicle_id} | {int(speed)} km/h - VIOLATION"
                color = (0, 0, 255)
            else:
                info_text = f"ID: {vehicle_id} | {int(speed)} km/h"
                color = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, info_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"ID:{vehicle_id}", (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f"Overspeed Cars: {overspeed_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Traffic Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()