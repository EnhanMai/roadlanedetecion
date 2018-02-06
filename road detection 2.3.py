import cv2
import numpy as np

import matplotlib.pyplot as plt

#selesct yellow and white color
def select_rgb_wy(image):
	lower = np.uint8([230,230,225]) #(G,B,R)
	upper = np.uint8([255,255,255])
	white_mask = cv2.inRange(image,lower,upper)

	lower = np.uint8([160,120,90]) #(G,B,R)
	upper = np.uint8([230,150,120])
	yellow_mask = cv2.inRange(image,lower,upper)

	mask = cv2.bitwise_or(yellow_mask,white_mask)
	masked = cv2.bitwise_and(image,image,mask = mask)
	return masked

#convert rgb into hsl
def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

#select white and yellow color in hsl
def select_hsl_wy(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([10, 80, 75]) #(G,B,R) 0 150 0
    upper = np.uint8([45, 255, 255]) #(G,B,R) 255 255 255
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10, 45,100]) # 10 0 70
    upper = np.uint8([ 22, 240,200]) # 40 255 255
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_smoothing(image, kernel_size=15):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image, low_threshold=40, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)


def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

    
def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]

    bottom_left  = [cols*0, rows*0.6]
    top_left     = [cols*0.1, rows*0.45]
    bottom_right = [cols*1, rows*0.6]
    top_right    = [cols*0.9, rows*0.45]

#    bottom_left  = [0, 612]
#    top_left     = [365, 377]
#    bottom_right = [1024, 768]
#    top_right    = [591, 350]

    
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

def hough_lines(image):
    """
    `image` should be the output of a Canny transform.
    
    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)


dis=300
def average_slope_intercept(lists):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)


    global dis

    for lines in lists:
         for line in lines:
                img= cv2.line(image,(line[0],line[1]),(line[2],line[3]),(255,0,0),5)
                print(line)
                print("distance=",_line2line(line))
                if _line2line(lines) < dis:     #can modify !!!
                        dis=_line2line(lines)
    for lines in lists:
            for line in lines:
                    


    
    for line in lists:
#        for line in lines:
            x1=line[0]
            y1=line[1]
            x2=line[2]
            y2=line[3]
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                  left_lines.append((slope, intercept))
                  left_weights.append((length))
            else:
                  right_lines.append((slope, intercept))
                  right_weights.append((length))
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    
    return left_lane, right_lane # (slope, intercept), (slope, intercept)

          
def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None
    
    slope, intercept = line
    
    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))
    
       


image = cv2.imread("15image.jpg")

(b,g,r)= cv2.split(image)
rgb_img = cv2.merge([b,g,r]) 

rgb_wy = select_rgb_wy(rgb_img)
hsl_wy = select_hsl_wy(rgb_img)

gray_images = convert_gray_scale(hsl_wy)

blurred_images=apply_smoothing(gray_images)
edge_image=detect_edges(blurred_images)

roi_images= select_region(edge_image)

list_of_lines = hough_lines( roi_images)
a,b= image.shape[:2]

print(a)
print(b)


def _line2line(line1):

    """
    - line1 is a list of two xy tuples
    - line2 is a list of two xy tuples
    References consulted:
    http://mathforum.org/library/drmath/view/51980.html
    and http://mathforum.org/library/drmath/view/51926.html
    and https://answers.yahoo.com/question/index?qid=20110507163534AAgvfQF
    """
    a=768
    b=1024
    import math
    #step1: cross prod the two lines to find common perp vector
    L1x1,L1y1,L1x2,L1y2 = line1
    L2x1,L2y1,L2x2,L2y2 =int(b/2) ,0 ,int(b/2) ,a 
    L1dx,L1dy = L1x2-L1x1,L1y2-L1y1
    L2dx,L2dy = L2x2-L2x1,L2y2-L2y1
    commonperp_dx,commonperp_dy = (L1dy - L2dy, L2dx-L1dx)

    #step2: normalized_perp = perp vector / distance of common perp
    commonperp_length = math.hypot(commonperp_dx,commonperp_dy)
    commonperp_normalized_dx = commonperp_dx/float(commonperp_length)
    commonperp_normalized_dy = commonperp_dy/float(commonperp_length)

    #step3: length of (pointonline1-pointonline2 dotprod normalized_perp).
    # Note: According to the first link above, it's sufficient to
    #    "Take any point m on line 1 and any point n on line 2."
    #    Here I chose the startpoint of both lines
    shortestvector_dx = (L1x1-L2x1)*commonperp_normalized_dx
    shortestvector_dy = (L1y1-L2y1)*commonperp_normalized_dy
    mindist = math.hypot(shortestvector_dx,shortestvector_dy)

    #return results
    result = mindist
    return result

new_lines=[]



img= cv2.line(image,(int(b/2),0),(int(b/2),a),(0,255,0),5)


left_lane, right_lane = average_slope_intercept(new_lines)

print("left:",left_lane)
print("right:", right_lane)

y1 = image.shape[0] 
y2 = y1*0.1 

new_left_line  = make_line_points(y1, y2, left_lane)
new_right_line = make_line_points(y1, y2, right_lane)




img= cv2.line(image,new_left_line[0] ,new_left_line[1] ,(0,0,255),10)
img= cv2.line(image,new_right_line[0] ,new_right_line[1] ,(0,0,255),10)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow("image",image)
cv2.waitKey(0)
