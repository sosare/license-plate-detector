import os
import cv2
import easyocr
from ultralytics import YOLO

def format_license_plate(ocr_results):
    """
    Analiza los resultados del OCR de forma dinámica.
    Selecciona la cadena de texto más larga y con mejor confianza.
    """
    if not ocr_results:
        return "", "NINGUNO"
        
    # Filtrar y limpiar candidatos
    candidates = []
    for (_, text, conf) in ocr_results:
        clean = text.strip().upper().replace(" ", "")
        # Ignorar ruidos de un solo carácter
        if len(clean) > 1:
            candidates.append((clean, conf))
            
    if not candidates:
        return "", "NINGUNO"
        
    # Seleccionar el candidato que tenga la cadena de texto más larga (habitualmente la placa principal)
    # En caso de empate en longitud, desempata por el que tenga mayor confianza
    best_candidate = max(candidates, key=lambda x: (len(x[0]), x[1]))
    main_block = best_candidate[0]
    
    # Recolectar textos adicionales si los hubiera
    extra_info = [text for (text, _) in candidates if text != main_block]
    extra_str = ", ".join(extra_info) if extra_info else "NINGUNO"
    
    return main_block, extra_str

def run_end_to_end_pipeline():
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    weights_path = os.path.join(project_root, "runs", "detect", "plate_detector", "weights", "best.pt")
    image_path = os.path.join(project_root, "test_car.jpg")
    
    if not os.path.exists(weights_path) or not os.path.exists(image_path):
        print("❌ Error: Verifica las rutas de 'best.pt' y 'test_car.jpg'")
        return

    # ==========================================
    # FASE 2: DETECCIÓN (YOLO)
    # ==========================================
    print("🤖 [1/3] Localizando placa del vehículo con YOLO...")
    model = YOLO(weights_path)
    img = cv2.imread(image_path)
    results = model.predict(source=img, conf=0.40, device="cpu", verbose=False)
    
    # Extraer el primer objeto de la lista de resultados
    first_result = results[0]
    
    if len(first_result.boxes) == 0:
        print("❌ Pipeline detenido: Placa no localizada.")
        return
        
    # Extraer las coordenadas de la primera caja detectada
    x1, y1, x2, y2 = map(int, first_result.boxes.xyxy[0].tolist())
    cropped = img[y1:y2, x1:x2]

    # ==========================================
    # FASE 3: PROCESAMIENTO DIGITAL Y OCR CON FALLBACK
    # ==========================================
    print("🛠️ [2/3] Generando matrices de imagen para el OCR...")
    # Creamos un escalado limpio a color directo (sin binarizar para conservar el contraste amarillo/negro nativo)
    resized_color = cv2.resize(cropped, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # También generamos una versión en escala de grises estándar por seguridad
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    print("🔤 [3/3] Extrayendo caracteres con EasyOCR (Estrategia de Fallback)...")
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Intento 1: Pasamos la imagen escalada a color nativo (ideal para contrastes como el de NY)
    ocr_results = reader.readtext(resized_color)
    plate, metadata = format_license_plate(ocr_results)
    
    # 🔄 FALLBACK: Si no leyó nada o la lectura es extremadamente corta, intenta con la escala de grises
    if len(plate) < 4:
        print("🔄 Alerta de baja lectura. Ejecutando canal secundario (Escala de Grises)...")
        ocr_results_backup = reader.readtext(resized_gray)
        plate_backup, metadata_backup = format_license_plate(ocr_results_backup)
        if len(plate_backup) > len(plate):
            plate, metadata = plate_backup, metadata_backup

    # ==========================================
    # FASE 4: SALIDA FINAL
    # ==========================================
    print("\n" + "="*40)
    print("🎯 RECONOCIMIENTO AUTOMATIZADO DE PUNTA A PUNTA")
    print("="*40)
    print(f"📌 TEXTO DETECTADO:  {plate if plate else 'NO LEÍDO'}")
    print(f"📌 TEXTOS EXTRAS:    {metadata}")
    print("="*40)

if __name__ == "__main__":
    run_end_to_end_pipeline()
