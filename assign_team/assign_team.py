
from sklearn.cluster import KMeans

import sys
sys.path.append('../')


class AssignTeam():
    def __init__(self):
        self.team_dict = {}
        self.team_color_dict = {}
    def clustering_player(self,image):
        image_2d=image.reshape(-1,3)
        
        #Preform K-means with 2 clusters
        kmeans=KMeans(n_clusters=2 , init='k-means++',n_init=10)
        kmeans.fit(image_2d)

        return kmeans

    def get_color_player(self,frame,bbox):
        images = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        top_half_pic = images[0:images.shape[0]//2,:]
        kmeans = self.clustering_player(top_half_pic)

        labels = kmeans.labels_
        cluster_image = labels.reshape(top_half_pic.shape[0],top_half_pic.shape[1])
        corner_cluster = [cluster_image[0,0],cluster_image[0,-1],cluster_image[-1,0],cluster_image[-1,-1]]
        non_player_cluster =max(set(corner_cluster), key=corner_cluster.count)


        player_cluster = 1- non_player_cluster
        player_color_center = kmeans.cluster_centers_[player_cluster]
        return player_color_center
        
    def assign_color_team(self,frame,players_detection):
        player_colors= []
        for track_id, detection in players_detection.items():
            bbox  =detection['bbox']
            color = self.get_color_player(frame,bbox)
            player_colors.append(color)
        kmeans = KMeans(n_clusters=2,n_init=1,init='k-means++')
        kmeans.fit(player_colors)
        self.kmeans = kmeans
        self.team_color_dict[1]=kmeans.cluster_centers_[0]
        self.team_color_dict[2]=kmeans.cluster_centers_[1]

        
    def get_color_team(self,frame,bbox,track_id):
        if track_id in self.team_dict :
            return  self.team_dict[track_id]
        color = self.get_color_player(frame,bbox)
        team_id =self.kmeans.predict(color.reshape(1,-1))[0]
        team_id +=1
        self.team_dict[track_id] = team_id
        return team_id
