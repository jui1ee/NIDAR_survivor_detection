import cv2
import time
from picamera2 import Picamera2
from ultralytics import YOLO

# camera setup
picam2 = Picamera2()

picam2.preview_configuration.main.size = (320, 320)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()

picam2.configure("preview")
picam2.start()
time.sleep(2)

#loading model
model = YOLO("best_v8n.pt")


#main loop
while True:
    frame = picam2.capture_array()
    
    #YOLO inference (CPU)
    results = model(frame, imgsz = 320, conf=0.40, verbose=False)
    annotated = results[0].plot()
    
    #printing detections to terminal
    boxes = results[0].boxes
    
    if boxes is not None and len(boxes) > 0:
        print("Detections:", len(boxes))
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"Class: {model.names[cls]}, Confidence: {conf: .2f}")
            
    
    #fps calculation
    infer_ms = results[0].speed["inference"]
    fps = 1000 / infer_ms if infer_ms > 0 else 0
    
    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
     )
    cv2.imshow("Team RESCUE",annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    #cleanup
    
picam2.stop()
cv2.destroyAllWindows()
    