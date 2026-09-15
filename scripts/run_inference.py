import os
import cv2
from ultralytics import YOLO

def force_fresh_inference():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    # Target our real working weights path
    weights_path = os.path.join(project_root, "runs", "detect", "plate_detector", "weights", "best.pt")     # "plate_detector-xxxx" modelo de pruebas que genera el run_inference.py
    image_path = os.path.join(project_root, "test_car.jpg")
    
    # 📌 CHANGE: Saving under a brand-new filename to completely bypass file cache locks
    output_path = os.path.join(project_root, "fresh_prediction_test.jpg")
    
    if not os.path.exists(weights_path) or not os.path.exists(image_path):
        print("❌ Error: Verification failed. Confirm 'test_car.jpg' or weights exist.")
        return

    print("🧠 Loading clean instance model weights...")
    model = YOLO(weights_path)
    
    # Read the car image using standard OpenCV (Keeps native color space locked)
    img = cv2.imread(image_path)
    
    print("📸 Testing at absolute zero threshold to force raw array detection data...")
    # conf=0.25 forces the engine to reveal any detected pixel coordinates
    results = model.predict(source=img, conf=0.25, device="cpu")
    boxes = results[0].boxes
    
    print(f"📊 Raw Model Coordinate Arrays found: {len(boxes)}")
    
    # If the model finds bounding boxes, we draw them ourselves natively
    if len(boxes) > 0:
        for box in boxes:
            # Extract bounding box boundaries in standard pixel coordinates (xyxy)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf_score = float(box.conf[0])
            
            # Draw a bright green rectangle outline natively over the BGR image array
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            
            # Write text label string above the bounding box
            label = f"Plate: {conf_score:.2f}"
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"✅ Drew box at coordinates: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}] with Conf: {conf_score:.4f}")
    else:
        print("⚠️ No geometric coordinates returned by the weights.")
        
    # Directly write out the native image file array
    cv2.imwrite(output_path, img)
    print(f"🎯 Brand new fresh image written to disk as: {output_path}")

if __name__ == "__main__":
    force_fresh_inference()