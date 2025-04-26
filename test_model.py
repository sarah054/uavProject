# from ultralytics import YOLO
#
# # טעינת המודל המוכן
# model = YOLO('runs/detect/train2/weights/best.pt')
#
# # מריצים את המודל על תמונה חדשה
# results = model(r'C:\Users\USER\Downloads\תמונה לבדיקה.jpg', save=True)
#
# # מציגים את התוצאות במסך
# results[0].show()

# הרצת המודל עם הצגת התמונה לפני הזיהוי
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


model = YOLO("runs/detect/train2/weights/best.pt")

# טען את התמונה בעזרת OpenCV
image_path = r"C:\אימון מודל\תמונה לבדיקה.jpg"
img = cv2.imread(image_path)

# המרת צבעים מ-BGR ל-RGB (כי OpenCV מציג בצבעים הפוכים)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# הצג את התמונה עם matplotlib
plt.imshow(img_rgb)
plt.title("התמונה לפני זיהוי")
plt.axis('off')
plt.show()

# שליחת התמונה לזיהוי
results = model(img, save=True)
print (results)

# ניקח את תיבת הזיהוי הראשונה
boxes = results[0].boxes

# קואורדינטות של התיבה
for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0]  # קצה שמאלי-עליון וקצה ימני-תחתון
    confidence = box.conf[0]      # רמת הביטחון בזיהוי (בין 0 ל-1)
    cls = box.cls[0]              # מספר הקטגוריה שזוהתה

    print(f"תיבת זיהוי: ({x1:.0f}, {y1:.0f}), ({x2:.0f}, {y2:.0f})")
    print(f"רמת ביטחון: {confidence:.2f}")
    print(f"קטגוריה: {cls}")

