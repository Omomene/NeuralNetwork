from ultralytics import YOLO
import shutil
import os

def build_yolo():

    model = YOLO("yolov8n.pt")

    return model


def train_yolo(
    data_yaml="dataset/furniture_detection/data.yaml",
    epochs=5
):

    model = build_yolo()

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=320,
        batch=16,
        project="runs",
        name="furniture_detector"
    )

    return model


def evaluate_yolo(
    model_path="runs/furniture_detector/weights/best.pt"
):

    model = YOLO(model_path)

    metrics = model.val()

    return metrics


def predict_image(
    image_path,
    model_path="runs/furniture_detector/weights/best.pt"
):

    model = YOLO(model_path)

    results = model.predict(
        source=image_path,
        conf=0.25,
        save=True
    )

    return results

def save_model(
    source_path="runs/furniture_detector/weights/best.pt",
    destination_path="models/yolo_furniture_detector.pt"
):

    os.makedirs("models", exist_ok=True)

    shutil.copy(
        source_path,
        destination_path
    )

    print(
        f"Model saved to {destination_path}"
    )