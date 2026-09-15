import os
from ultralytics import YOLO

def train_license_plate_detector():
    # Encuentra la ubicación exacta de este archivo (data.yaml)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    yaml_path = os.path.join(project_root, "data", "dataset", "data.yaml")
    
    print(f"🚀 Cargando la configuración de: {yaml_path}")
    
    # Cargar un modelo pre-entrenadoYOLOv8 Nano (Transfer Learning)
    # Modelo capaz de identificar formas basicas y bordes.
    model = YOLO("yolov8n.pt")
    
    print("🏋️ Iniciando el proceso de entrenamiento en CPU...")
    
    # Executar la rutina de entrenamiento optimizada para CPU
    model.train(
        data=yaml_path,         # Ruta a la configuración de datos
        epochs=20,              # Número de rondas de entrenamiento (optimizado para la velocidad de CPU)
        imgsz=640,              # Tamaño estándar de resolución de imagen (640x640)
        device="cpu",           # Forzar la ejecución del procesamiento estrictamente en CPU
        workers=2,              # Limitar los hilos de carga de datos en segundo plano para evitar fallos
        patience=35,            # Paciencia para detener el entrenamiento si no hay mejora
        name="plate_detector"   # Nombre de la carpeta del directorio de salida
    )
    
    print("🎉 Proceso de entrenamiento completado con éxito!")

if __name__ == "__main__":
    train_license_plate_detector()
