from ultralytics import YOLO

model = YOLO(r"C:\Users\Public\Documents\Football_Project\runs\detect\train\weights\best.pt")  
results = model.predict(source=r"C:\Users\Public\Documents\Football_Project\input_videos\08fd33_4.mp4", conf=0.4, save=True)
print(results)