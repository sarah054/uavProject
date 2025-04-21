from ultralytics import YOLO

# שלב 1: טעינת המודל (מודל YOLOv8 משמש לזיהוי)
model = YOLO("yolov8n.pt")  # אפשר גם לנסות את yolov8s.pt או yolov8m.pt למודלים חזקים יותר

# שלב 2: הרצת האימון
model.train(
    data="data.yaml",  # הנתיב לקובץ ה-YAML
    epochs=50,         # מספר האפוקים - אפשר להגדיל אם יש צורך בדיוק גבוה יותר
    batch=8,           # מספר הדוגמאות בכל שלב אימון (תלוי בכוח המחשב שלך)
    imgsz=640,         # גודל התמונות באימון
    device="cpu"      # השתמשי ב-GPU אם יש לך, אחרת ניתן לשים "cpu"
)

