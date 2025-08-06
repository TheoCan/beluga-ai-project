from ultralytics import YOLO
import cv2
import pandas as pd
import os

# Load your trained model
model = YOLO("runs/detect/train4/weights/best.pt")  # Use your custom-trained beluga detector

# Load video
video_path = 'videos/beluga1_video.mp4'  # Update with your actual video name
cap = cv2.VideoCapture(video_path)

# Check if video opened correctly
if not cap.isOpened():
    print("⚠️ Could not open video.")
    exit()

positions = []
frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: Resize to match training resolution
    frame = cv2.resize(frame, (640, 640))

    # Predict with lower confidence threshold (e.g., 0.3)
    results = model.predict(source=frame, conf=0.7, verbose=False)

    boxes = results[0].boxes
    print(f"Frame {frame_num}: {len(boxes)} detections")

    for box in boxes:
        # Get box coordinates and confidence
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()

        # Draw rectangle
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"Beluga {conf:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save center point and confidence
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        positions.append((frame_num, center_x, center_y, conf))

    # Optional: display the video as it runs
    cv2.imshow('Beluga YOLO Tracker', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    frame_num += 1

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Save results only if there are detections
if positions:
    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(positions, columns=['frame', 'x', 'y', 'confidence'])
    df.to_csv('outputs/beluga_path_yolo.csv', index=False)
    print(f"✅ YOLO Tracking complete. Saved {len(df)} detections to: outputs/beluga_path_yolo.csv")
else:
    print("⚠️ No detections were made. CSV was not saved.")

print(f"🎬 Total frames processed: {frame_num}")
print(f"📍 Total beluga detections: {len(positions)}")
