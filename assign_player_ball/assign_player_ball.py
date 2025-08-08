from utils import get_center_bbox,measure_distance
import sys
sys.path.append('../')

class AssignPlayer():
    def __init__(self):
        self.max_distance = 700
        
    def get_player_have_ball(self,players,ball_bbox):
        x_center_ball,y_center_ball = get_center_bbox(ball_bbox)
        minimum_distance = 999
        assign_player = - 1

        for track_id,player in players.items():
            bbox_player = player['bbox']
            left = measure_distance((bbox_player[0],bbox_player[3]),(x_center_ball,y_center_ball))
            right =  measure_distance((bbox_player[1],bbox_player[3]),(x_center_ball,y_center_ball))
            distance_min = min(left,right)

            if distance_min <self.max_distance:
                if distance_min < minimum_distance:
                    minimum_distance=distance_min
                    assign_player=track_id
        return assign_player

