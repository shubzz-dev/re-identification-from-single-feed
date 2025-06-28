import cv2
import time
import cvzone
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# to load your custom-trained YOLOv8 model
model = YOLO("topic2_model.pt")  # or "best.pt"

#to load the input video
cap = cv2.VideoCapture("15sec_input_720p - Copy.mp4")

if not cap.isOpened():
    print("Error: Could not open video file!")
    exit()

# Get FPS and wait time
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / fps) if fps != 0 else 33

# to initialize DeepSORT tracker with appearance feature extractor
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.4,
    embedder="mobilenet",  # uses visual features for re-ID
    half=True              # for faster CPU inference
)

# getting  model class names
classnames = model.names
print(" Model Classes:", classnames)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = []

    # Resize for faster inference
    resized = cv2.resize(frame, (640, 360))
    results = model(resized)[0]

    scale_x = frame.shape[1] / 640
    scale_y = frame.shape[0] / 360

    # loop to go through each detection
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = classnames[cls]

        print(f"Detected: {class_name} | Confidence: {conf:.2f}")

        if conf > 0.15:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            w, h = x2 - x1, y2 - y1


            detections.append(([x1, y1, w, h], conf, class_name))

            #to  Draw detection before tracking
            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorC=(255, 0, 255))
            cvzone.putTextRect(frame, f"{class_name}", (x1, y1 - 10), scale=1, thickness=1)



    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        x1, y1, x2, y2 = int(l), int(t), int(r), int(b)
        w, h = x2 - x1, y2 - y1

        # to Draw re-ID box with ID label
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorC=(0, 255, 0))
        cvzone.putTextRect(frame, f"ID: {track_id}", (x1, y1 - 30), scale=1, thickness=2, colorT=(255, 255, 255), colorR=(0, 255, 0))

    # for Show frame
    cv2.imshow(" Player Re-Identification ~", frame)
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()