import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("PPE.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame, agnostic_nms=True)[0]

    if not results or len(results) == 0:
        continue

    for result in results:

        detection_count = result.boxes.shape[0]

        for i in range(detection_count):
            cls = int(result.boxes.cls[i].item())
            name = result.names[cls]
            confidence = float(result.boxes.conf[i].item())
            bounding_box = result.boxes.xyxy[i].cpu().numpy()

            x = int(bounding_box[0])
            y = int(bounding_box[1])
            width = int(bounding_box[2] - x)
            height = int(bounding_box[3] - y)
            color = g_c(name)
            
            cv2.rectangle(frame, (x, y), (x + width, y + height), (color), 2)
            cv2.rectangle(frame, (x, y-30), (x + width, y), (color), -1)

            # Display class and confidence
            text = f"{name}: {confidence:.2f}%"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)
            if name in ['No-Helmet', 'No-Gloves', 'No-Vest']:
                print(f"ALERT: Person detected without {name}!")


    if frame.shape[0] > 0 and frame.shape[1] > 0:
        # Display the resulting frame
        cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
def g_c(id):
    np.random.seed(10)
    colors = np.random.randint(100, 255, size=3).tolist()
    return tuple(colors[id])
