import os
import cv2
import easyocr

def format_license_plate(ocr_results):
    """
    Analiza los resultados del OCR utilizando la lógica exacta de la ruta modular.
    Selecciona el candidato con la cadena de texto más larga.
    """
    if not ocr_results:
        return "", "NINGUNO"
        
    candidates = []
    for (_, text, conf) in ocr_results:
        clean = text.strip().upper().replace(" ", "")
        if len(clean) > 1:
            candidates.append((clean, conf))
            
    if not candidates:
        return "", "NINGUNO"
        
    # Selección limpia basada estrictamente en la longitud del texto (Idéntico a main.py)
    best_candidate = max(candidates, key=lambda x: len(x[0]))
    main_block = best_candidate[0]
    
    extra_info = [text for (text, _) in candidates if text != main_block]
    extra_str = ", ".join(extra_info) if extra_info else "NINGUNO"
    
    return main_block, extra_str

def run_optimized_ocr():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    crop_path = os.path.join(project_root, "cropped_plate.jpg")
    
    if not os.path.exists(crop_path):
        print("❌ Error: No se encuentra cropped_plate.jpg")
        return

    # 1. Cargar la imagen original
    img = cv2.imread(crop_path)
    
    # 2. 🛠️ PROCESAMIENTO DIGITAL DE IMÁGENES (Alineado con main.py)
    # Escalado limpio a color directo
    resized_color = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 📌 CORRECCIÓN: Pasar a escala de grises desde resized_color para igualar a main.py
    gray = cv2.cvtColor(resized_color, cv2.COLOR_BGR2GRAY)
    
    # Segundo escalado e interpolación idéntica a tu main.py
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
   
    print("🧠 Inicializando EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False)
    
    print("🔤 Ejecutando lectura sobre la imagen a color...")
    # Intento 1: Pasamos la imagen escalada a color nativo
    ocr_results = reader.readtext(resized_color)
    
    # 📌 CORRECCIÓN: Capturar las variables locales del primer intento de lectura
    plate, metadata = format_license_plate(ocr_results)
    
    if len(ocr_results) > 0:
        print("\n📋 Resultados del Intento a Color:")
        print("-" * 40)
        for bounding_box, text, confidence in ocr_results:
            clean_text = text.strip().upper()
            clean_text = clean_text.replace(":", "-").replace("*", "-")
            print(f"🔤 Texto: '{clean_text}' | Confianza: {confidence:.2f}")
        print("-" * 40)
    else:
        print("⚠️ No se detectó texto en el canal a color.")

    # 🔄 FALLBACK IDÉNTICO A MAIN.PY: Evaluación del tamaño de la cadena rescatada
    if len(plate) < 4:
        print("🔄 Alerta de baja lectura. Ejecutando canal secundario (Escala de Grises)...")
        ocr_results_backup = reader.readtext(resized)
        plate_backup, metadata_backup = format_license_plate(ocr_results_backup)
        if len(plate_backup) > len(plate):
            plate, metadata = plate_backup, metadata_backup

    # Despliegue final de control de las variables formateadas
    print(f"\n🎯 Texto Final Consolidado: {plate}")
    print(f"📌 Metadatos Extra: {metadata}")

if __name__ == "__main__":
    run_optimized_ocr()
