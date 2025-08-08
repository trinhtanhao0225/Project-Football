def get_center_bbox(bbox):
    x1,y1,x2,y2=bbox
    x_center = (x1+x2)/2
    y_center = (y1+y2)/2

    return x_center,y_center
def get_width_height_bbox(bbox):
    x1,y1,x2,y2=bbox
    width = x2-x1
    height = y2-y1
    return width,height

def measure_distance(p1,p2):

    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5