from utils import save_video,read_video
from tracker import Tracker
import os
from assign_team import AssignTeam
from assign_player_ball import AssignPlayer
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def main():
    frames = read_video(r'C:\Users\Public\Documents\Football_Project\input_videos\08fd33_4.mp4')
    trackers = Tracker(r"C:\Users\Public\Documents\Football_Project\runs\detect\train\weights\best.pt")
    tracks=trackers.get_object_by_tracker(frames,True,'tracker/tracks.pkl') 
    tracks['ball']=trackers.interpolate_lball_positions(tracks['ball'])


    team = AssignTeam()
    team.assign_color_team(frames[0],tracks['player'][0])

    player_have_ball = AssignPlayer()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['player']):

        for player_id,track in player_track.items():
            team_id=team.get_color_team(frames[frame_num],track['bbox'],player_id)
            tracks['player'][frame_num][player_id]['team']=team_id
            tracks['player'][frame_num][player_id]['team_color']=team.team_color_dict[team_id]

               
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        player_id = player_have_ball.get_player_have_ball(tracks['player'][frame_num],ball_bbox) 


        if player_id != -1:
            tracks['player'][frame_num][player_id]['has_ball']= True
            team_ball_control.append(tracks['player'][frame_num][player_id]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)



    output_frames = trackers.draw_annotaion(frames,tracks,team_ball_control)
    save_video(output_frames,r'output_videos\output.avi')

if __name__=='__main__':
    main()