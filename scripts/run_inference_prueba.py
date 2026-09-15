import os
import cv2
from ultralytics import YOLO


def force_fresh_inference():

    # ---------------------------------------------------------
    # 1. Definir rutas del proyecto
    # ---------------------------------------------------------

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    weights_path = os.path.join(
        project_root,
        "runs",
        "detect",
        "plate_detector",
        "weights",
        "best.pt"
    )

    image_path = os.path.join(
        project_root,
        "test_car.jpg"
    )

    output_path = os.path.join(
        project_root,
        "fresh_prediction_test.jpg"
    )

    # ---------------------------------------------------------
    # 2. Verificar archivos
    # ---------------------------------------------------------

    if not os.path.exists(weights_path):
        print("❌ Error: No se encontró best.pt")
        print(f"   Ruta buscada: {weights_path}")
        return

    if not os.path.exists(image_path):
        print("❌ Error: No se encontró test_car.jpg")
        print(f"   Ruta buscada: {image_path}")
        return

    # ---------------------------------------------------------
    # 3. Cargar modelo
    # ---------------------------------------------------------

    print("🧠 Loading clean instance model weights...")
    print(f"📦 Weights: {weights_path}")

    model = YOLO(weights_path)

    print(f"🏷️ Model classes: {model.names}")

    # ---------------------------------------------------------
    # 4. Leer imagen
    # ---------------------------------------------------------

    img = cv2.imread(image_path)

    if img is None:
        print("❌ Error: OpenCV no pudo leer la imagen.")
        return

    print(f"📸 Image loaded: {image_path}")

    # ---------------------------------------------------------
    # 5. Prueba automática de diferentes confidence thresholds
    # ---------------------------------------------------------

    confidence_thresholds = [
        0.01,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50
    ]

    print()
    print("=" * 60)
    print("CONFIDENCE THRESHOLD TEST")
    print("=" * 60)

    detection_results = []

    for conf_threshold in confidence_thresholds:

        results = model.predict(
            source=img,
            conf=conf_threshold,
            device="cpu",
            verbose=False
        )

        boxes = results[0].boxes

        number_of_detections = len(boxes)

        # Obtener la confianza máxima encontrada
        if number_of_detections > 0:
            confidence_values = boxes.conf.tolist()
            max_confidence = max(confidence_values)
        else:
            max_confidence = None

        detection_results.append(
            (
                conf_threshold,
                number_of_detections,
                max_confidence
            )
        )

        if max_confidence is not None:
            print(
                f"conf={conf_threshold:.2f} "
                f"→ {number_of_detections} detección(es) "
                f"| confianza máxima={max_confidence:.4f}"
            )
        else:
            print(
                f"conf={conf_threshold:.2f} "
                f"→ 0 detecciones"
            )

    # ---------------------------------------------------------
    # 6. Identificar la confianza máxima real del modelo
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("RESULTADO DEL DIAGNÓSTICO")
    print("=" * 60)

    all_confidences = [
        result[2]
        for result in detection_results
        if result[2] is not None
    ]

    if not all_confidences:

        print("❌ El modelo no produjo ninguna detección.")
        print("   Debemos revisar el entrenamiento/dataset.")

        return

    maximum_confidence = max(all_confidences)

    print(
        f"🎯 Confianza máxima encontrada: "
        f"{maximum_confidence:.4f}"
    )

    # ---------------------------------------------------------
    # 7. Inferencia final para visualizar la detección
    # ---------------------------------------------------------
    #
    # Para esta prueba utilizamos 0.10 porque actualmente
    # sabemos que ese umbral produce detecciones.
    #
    # Esto es solamente para visualización/diagnóstico.
    # ---------------------------------------------------------

    visualization_conf = 0.10

    print()
    print(
        f"🖼️ Generando imagen utilizando conf="
        f"{visualization_conf:.2f}"
    )

    results = model.predict(
        source=img,
        conf=visualization_conf,
        device="cpu",
        verbose=False
    )

    boxes = results[0].boxes

    print(
        f"📊 Detecciones encontradas: {len(boxes)}"
    )

    # ---------------------------------------------------------
    # 8. Dibujar las detecciones
    # ---------------------------------------------------------

    if len(boxes) > 0:

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            conf_score = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                3
            )

            label = (
                f"{class_name}: "
                f"{conf_score:.2f}"
            )

            cv2.putText(
                img,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            print(
                f"✅ Detection:"
                f" [{int(x1)}, {int(y1)}, "
                f"{int(x2)}, {int(y2)}]"
                f" | Class: {class_name}"
                f" | Conf: {conf_score:.4f}"
            )

    else:

        print(
            "⚠️ No se encontraron detecciones "
            f"con conf={visualization_conf:.2f}"
        )

    # ---------------------------------------------------------
    # 9. Guardar imagen
    # ---------------------------------------------------------

    cv2.imwrite(output_path, img)

    print()
    print(
        f"🎯 Imagen de resultado guardada en:"
        f"\n   {output_path}"
    )


if __name__ == "__main__":

    force_fresh_inference()