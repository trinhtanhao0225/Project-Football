from ultralytics import YOLO
import supervision as sv
import pickle as pk
import os
import sys
import cv2
import numpy as np
import pandas as pd
sys.path.append('../')
from utils import get_center_bbox,get_width_height_bbox
class Tracker():
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    def get_frame_batch_size(self,frames):
        batch_size=20
        batch_frame = []
        for cnt_frame in range(0,len(frames),batch_size):
            batch_frame+=self.model.predict(frames[cnt_frame:cnt_frame+batch_size],conf = 0.5)
        return batch_frame

    def get_object_by_tracker(self,frames,use_cache=True,cache_path=None):
  
        if use_cache is True and os.path.exists(cache_path):
            with open(cache_path,'rb') as f :
                trackers =pk.load(f)
            return trackers

        detections = self.get_frame_batch_size(frames)

        tracks = {
            'player' : [],
            'ball' : [],
            'referee' : []
        }

        for frame_num,detection in enumerate(detections):
            name = detection.names
            name_id = {v:k for k,v in name.items()}
            detection_vision =sv.Detections.from_ultralytics(detection)

            
            for object_key,object_value in enumerate(detection_vision.class_id):
                if name[object_value]=='goalkeeper':
                    detection_vision.class_id[object_key] = name_id['player']
            detection_with_tracks = self.tracker.update_with_detections(detection_vision)

            tracks['player'].append({})
            tracks['ball'].append({})
            tracks['referee'].append({})

            for frame_detect in detection_with_tracks:
                bbox = frame_detect[0]
                class_id =frame_detect[3]
                track_id = frame_detect[4]

                if class_id == name_id['player']:
                    tracks['player'][frame_num][track_id] = {'bbox':bbox}

                if class_id == name_id['referee']:
                    tracks['referee'][frame_num][track_id] = {'bbox':bbox}

            for frame_detection in detection_vision:
                bbox =frame_detection[0].tolist()
                class_id = frame_detection[3]
                if class_id == name_id['ball']:
                    tracks['ball'][frame_num][1] = {'bbox':bbox}

        with open(cache_path,'wb') as f:
            pk.dump(tracks,f)

        return tracks
    
    def interpolate_lball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions=df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:{'bbox':x} }for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        #Draw a semi-transparent rectaggle

        overlay=frame.copy()
        cv2.rectangle(overlay,(1350,850),(1900,970),(255,255,255),-1)
        alpha = 0.4
        cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        #Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]

        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame,f'Team 1 ball control: {team_1*100:.2f}%' ,(1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv2.putText(frame,f'Team 2 ball control: {team_2*100:.2f}%' ,(1400,930),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

        return frame
    def draw_ellipse(self, frame, bbox, color, track_id):

        y2 = int(bbox[3])
        x_center, _ = get_center_bbox(bbox)
        width, _ = get_width_height_bbox(bbox)

        center = (int(x_center), y2)
        axes = (int(width), int(0.35 * width))

        # ⚠️ Không truyền theo tên đối số để tránh lỗi OpenCV
        cv2.ellipse(
            frame,
            center,
            axes,
            0.0,       # angle
            -45,       # startAngle
            215,       # endAngle
            color,
            2,         # thickness
            cv2.LINE_4 # lineType
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2

        y1_rect = (y2 - rectangle_height//2 ) +15
        y2_rect = (y2 + rectangle_height//2 ) +15
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED,

                          )
            x1_text=x1_rect + 20
            if track_id >99:
                x1_text-=10
            cv2.putText(
                frame,
                f'{track_id}',
                (int(x1_rect),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )
        return frame
    def draw_traingle(self, frame, bbox, color):
        if bbox is None or len(bbox) != 4:
            return frame  # bỏ qua nếu bbox không hợp lệ

        x_center, _ = get_center_bbox(bbox)
        y = int(bbox[1])

        # Kiểm tra x_center hợp lệ
        if x_center is None:
            return frame

        triangle_points = np.array([
            [int(x_center), int(y)],
            [int(x_center) - 10, int(y) - 20],
            [int(x_center) + 10, int(y) - 20],
        ], dtype=np.int32).reshape((-1, 1, 2))

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotaion(self,frames,tracks,team_ball_control):
        output_frames = []
        for frame_id,frame in enumerate(frames):
            frame=frame.copy()
            dict_player = tracks['player'][frame_id]
            dict_referee = tracks['referee'][frame_id]
            dict_ball = tracks['ball'][frame_id]

            for track_id,player in dict_player.items():
                color = player['team_color']
                frame = self.draw_ellipse(frame,player['bbox'],color,track_id)

                if player.get('has_ball',True):
                    frame=self.draw_traingle(frame,player['bbox'],(0,0,255))
                

            for track_id,referee in dict_referee.items():
                frame = self.draw_ellipse(frame,referee['bbox'],(0,255,0),None)
            
            for track_id, ball in dict_ball.items():
                bbox = ball.get('bbox')
                
                frame = self.draw_traingle(frame, bbox, (0, 0, 255))

            frame = self.draw_team_ball_control(frame,frame_id,team_ball_control)

            output_frames.append(frame)
        return output_frames