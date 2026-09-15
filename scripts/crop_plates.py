import os
import cv2
from ultralytics import YOLO

def run_production_crop():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    # Crear una variable para la carpeta de pesos, permitiendo cambiar entre diferentes modelos entrenados
    weights_folder = "plate_detector"  # Ejemplo: "plate_detector-4" o "plate_detector-5"
    weights_path = os.path.join(project_root, "runs", "detect", weights_folder, "weights", "best.pt")
    
    image_path = os.path.join(project_root, "test_car.jpg")
    crop_output_path = os.path.join(project_root, "cropped_plate.jpg")
    
    # Validaciones de archivos obligatorias
    if not os.path.exists(weights_path):
        print(f"❌ Error: No se encontraron los pesos en: {weights_path}")
        return
    if not os.path.exists(image_path):
        print(f"❌ Error: No se encontró la imagen de prueba 'test_car.jpg' en la raíz.")
        return

    # 1. Cargar imagen de manera nativa con OpenCV (Formato BGR estándar)
    img = cv2.imread(image_path)
    
    print(f"🧠 Cargando modelo de alta confianza desde: {weights_folder}")
    model = YOLO(weights_path)
    
    print("📸 Ejecutando inferencia de producción a un umbral exigente del 50%...")
    # Filtramos con conf=0.50 porque sabemos que tu modelo supera el 0.80 con soltura
    results = model.predict(source=img, conf=0.50, device="cpu", verbose=False)
    first_result = results[0]
    boxes = first_result.boxes
    
    if len(boxes) > 0:
        # Tomamos la caja con mayor confianza absoluta
        best_box = boxes[0]
        conf_score = float(best_box.conf[0])
        
        # 2. Extraer las coordenadas exactas mapeadas a enteros (píxeles)
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        
        print(f"🏆 ¡Placa localizada! Confianza: {conf_score:.2f}%")
        print(f"📍 Coordenadas de corte geométrico: X=[{x1}:{x2}], Y=[{y1}:{y2}]")
        
        # 3. Slicing matricial de NumPy: [Y_inicial : Y_final, X_inicial : X_final]
        # Corta la imagen exactamente dentro de los límites del rectángulo
        cropped_plate = img[y1:y2, x1:x2]
        
        if cropped_plate.size > 0:
            # Guardar el recorte limpio
            cv2.imwrite(crop_output_path, cropped_plate)
            print(f"🎯 Recorte guardado con éxito como: {crop_output_path}")
        else:
            print("⚠️ El arreglo del recorte resultó vacío.")
    else:
        print("❌ El modelo no detectó ninguna placa con un umbral superior al 50%. Verifica la imagen de prueba.")

if __name__ == "__main__":
    run_production_crop()
