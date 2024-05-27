import cv2
import numpy as np
from ultralytics import YOLO
import pygame
import time
from twilio.rest import Client

pygame.mixer.init()

alert_sound = pygame.mixer.Sound('1709563840214ccwr0xsh-voicemaker.in-speech.wav')
last_alert_time = 0

account_sid = 'your_account_sid'
auth_token = 'your_auth_token'
twilio_phone_number = 'your_twilio_phone_number'
your_phone_number = 'your_phone_number'

twilio_client = Client(account_sid, auth_token)

def generate_random_color(id):
    np.random.seed(10)
    colors = np.random.randint(100, 255, size=(100, 3)).tolist()
    color_id = id % len(colors)
    return tuple(colors[color_id])

model = YOLO("best (2).pt")

video_file = 'video_file_path.mp4'  # Change this to your video file path

c = cv2.VideoCapture(video_file)

while True:
    ret, fr = c.read()
    if not ret:
        break

    rts = model(fr, agnostic_nms=True)[0]

    if not rts or len(rts) == 0:
        continue

    for rt in rts:
        dc = rt.boxes.shape[0]

        for i in range(dc):
            ci = int(rt.boxes.cls[i].item())
            cn = rt.names[ci]
            cf = float(rt.boxes.conf[i].item())
            bb = rt.boxes.xyxy[i].cpu().numpy()

            x = int(bb[0])
            y = int(bb[1])
            w = int(bb[2] - x)
            h = int(bb[3] - y)
            color = generate_random_color(ci)

            cv2.rectangle(fr, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(fr, (x, y-30), (x + w, y), color, -1)
            t = f"{cn}: {cf:.2f}%"
            cv2.putText(fr, t, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

            if cn in ['Tiger', 'Elephant', 'Cheetah','Bear']:
                current_time = time.time()
                if current_time - last_alert_time > 60:
                    print(f"ALERT: Animal detected: {cn}!")
                    alert_sound.play()
                    twilio_client.messages.create(
                        body=f"ALERT: Animal detected: {cn}!",
                        from_=twilio_phone_number,
                        to=your_phone_number
                    )
                    last_alert_time = current_time

    if fr.shape[0] > 0 and fr.shape[1] > 0:
        cv2.imshow('Animal Monitor', fr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

c.release()
cv2.destroyAllWindows()