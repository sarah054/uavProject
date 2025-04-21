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
image_path = r"C:\אימון מודל\תמונה1.png"
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
# list2 = results[0].boxes
# for item in list2:
#     print(item)


# from ultralytics import YOLO
#
# # טוענים את המודל המאומן
# model = YOLO("runs/detect/train2/weights/best.pt")
#
# # מבצעים הערכת ביצועים על סט ה-validation
# metrics = model.val()
#
# # מדפיסים את התוצאות
# print(metrics)
#בדיקה-----------