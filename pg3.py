import math




def calculate_line_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

def calculate_distance_between_point_and_line(x, y, x1, y1, x2, y2):
    m = calculate_line_slope(x1,y1,x2,y2)
    f = (m*x) - (m*x2) - y + y2
    s = math.sqrt(m**2 + 1)    
    return f / s



print(calculate_distance_between_point_and_line(520, 490, 470,500, 600,440))
print(calculate_distance_between_point_and_line(480, 450, 470,500, 600,440))
print(calculate_distance_between_point_and_line(600, 470, 470,500, 600,440))
print(calculate_distance_between_point_and_line(600, 430, 470,500, 600,440))






