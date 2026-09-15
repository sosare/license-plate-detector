import os
import cv2
import easyocr

def run_optimized_ocr():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    crop_path = os.path.join(project_root, "cropped_plate.jpg")
    
    if not os.path.exists(crop_path):
        print("❌ Error: No se encuentra cropped_plate.jpg")
        return

    # 1. Cargar la imagen original
    img = cv2.imread(crop_path)
    
    # 2. 🛠️ PROCESAMIENTO DIGITAL DE IMÁGENES (Fase [F3-P3])
    # Paso A: Pasar a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Paso B: Aumentar el tamaño (Upscaling) para que los caracteres pequeños sean más legibles
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # AÑADIR UN BLUR SUAVE: Rompe la fusión de píxeles negros vecinos
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    
    # Paso C: Aplicar Binarización Gaussiana (Otsu Thresholding)
    # Convierte la placa a blanco y negro puro, eliminando variaciones de luz naranja
    _, binary_plate = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Opcional: Guardar la imagen preprocesada para que puedas ver cómo la ve el OCR
    debug_ocr_path = os.path.join(project_root, "binarized_plate_debug.jpg")
    cv2.imwrite(debug_ocr_path, binary_plate)
    
    print("🧠 Inicializando EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False)
    
    print("🔤 Ejecutando lectura sobre la imagen binarizada...")
    # Pasamos la matriz procesada 'binary_plate' en lugar de la imagen cruda
    ocr_results = reader.readtext(binary_plate)
    
    if len(ocr_results) > 0:
        print("\n📋 Nuevos Resultados Optimizados:")
        print("-" * 40)
        for bounding_box, text, confidence in ocr_results:
            clean_text = text.strip().upper()
            
            # Limpieza post-procesamiento opcional por código:
            # Reemplazar los errores comunes de símbolos si persisten
            clean_text = clean_text.replace(":", "-").replace("*", "-")
            
            print(f"🔤 Texto: '{clean_text}' | Confianza: {confidence:.2f}")
        print("-" * 40)
    else:
        print("⚠️ No se detectó texto con los filtros aplicados.")

if __name__ == "__main__":
    run_optimized_ocr()
