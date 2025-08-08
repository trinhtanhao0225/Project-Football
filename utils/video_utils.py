
import cv2
def read_video(video_path):
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    frames = []
    while True :
        flag,frame=cap.read()
        if not flag:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out =cv2.VideoWriter(output_video_path,fourcc,25.0,(output_video_frames[0].shape[1],output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()