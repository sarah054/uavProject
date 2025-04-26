# import cv2
# from ultralytics import YOLO
#
# # טען את המודל
# model = YOLO("runs/detect/train2/weights/best.pt")
#
# # פתיחת קובץ הווידאו
# video_path = r"C:\אימון מודל\20250402-2005-17.2223385.mp4"
# vidObj = cv2.VideoCapture(video_path)
#
# # משתנים
# frame_rate = 0.5  # כמה פריימים לשנייה לקחת (כאן לוקחים אחד כל חצי שנייה)
# batch_size = 1  # כמה תמונות לשלוח למודל (לשפר מהירות)
# frames = []
# count = 0
#
# while True:
#     success, frame = vidObj.read()
#     if not success:
#         break
#
#     # לבדוק כל frame_rate פריימים
#     if int(vidObj.get(cv2.CAP_PROP_POS_FRAMES)) % (int(vidObj.get(cv2.CAP_PROP_FPS)) * frame_rate) == 0:
#         # המרת צבעים ל-RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(frame_rgb)
#
#     # כל batch_size תמונות - שלח למודל
#     if len(frames) == batch_size:
#         results = model(frames, save=False)  # save=False שלא ישמור קבצים פיזיים
#
#         # עיבוד תוצאות
#         for result in results:
#             if not result.boxes:
#                 print("No detections in current frame batch")
#             for box in result.boxes:
#                 cls_id = int(box.cls[0])
#                 confidence = box.conf[0]
#                 print(f"זיהוי: {model.names[cls_id]}, ביטחון: {confidence:.2f}")
#
#         frames = []  # אפס את הרשימה אחרי שליחה
#
# # במקרה שנשארו תמונות בסוף
# if frames:
#     results = model(frames, save=False)
#     for result in results:
#         if not result.boxes:
#             print("No detections in remaining frames")
#         for box in result.boxes:
#             cls_id = int(box.cls[0])
#             confidence = box.conf[0]
#             print(f"זיהוי: {model.names[cls_id]}, ביטחון: {confidence:.2f}")
#
# # שחרור משאבים
# vidObj.release()


import cv2
from ultralytics import YOLO

# טען את המודל
model = YOLO("runs/detect/train2/weights/best.pt")

# פתח את קובץ הוידאו
video_path = r"C:\אימון מודל\20250402-2005-17.2223385.mp4"
vidObj = cv2.VideoCapture(video_path)

# קבל פרטים על הוידאו
frame_width = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vidObj.get(cv2.CAP_PROP_FPS)

# הגדר קובץ פלט
out = cv2.VideoWriter('output_with_detections.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (frame_width, frame_height))

while True:
    success, frame = vidObj.read()
    if not success:
        break

    # רץ זיהוי על הפריים
    results = model(frame)

    # עבור כל זיהוי בפריים
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            confidence = box.conf[0]

            if confidence > 0.6:  # סף ביטחון
                # צייר תיבה
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[cls_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

    # הראה את הפריים
    cv2.imshow("Detections", frame)

    # כתוב את הפריים לקובץ הפלט
    out.write(frame)

    # יציאה אם לוחצים על 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# סגור הכל
vidObj.release()
out.release()
cv2.destroyAllWindows()
