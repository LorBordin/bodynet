from pickle import REDUCE
import cv2

RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE =  (255, 255, 255)

def draw_keypoints(img, coords, thresh, keypoints):
    """
    Draws the skeleton of a person. Keypoints below a given threshold are ignored.

    Parameters
    ----------
    img : np.array (dtype: uint8)
        Input image.
    coords : np.array (dtype: float)
        Keypoints prediction coordinates and scores in format (y, x, p).
    thresh : float
        Confidence score threshold.
    keypoints : dict
        Keypoints names and position in the coords array.

    Returns
    -------
    img : np.array (dtype: uint8)
        Output image. 
    """
    # head
    img = draw_bone(img, coords[keypoints["left_eye"]], coords[keypoints["left_ear"]], thresh,  color=RED)    
    img = draw_bone(img, coords[keypoints["left_eye"]], coords[keypoints["nose"]], thresh,  color=RED)

    img = draw_bone(img, coords[keypoints["right_eye"]], coords[keypoints["right_ear"]], thresh,  color=GREEN)
    img = draw_bone(img, coords[keypoints["right_eye"]], coords[keypoints["nose"]], thresh,  color=GREEN)
    
    # arms
    img = draw_bone(img, coords[keypoints["left_shoulder"]], coords[keypoints["left_elbow"]], thresh,  color=RED)
    img = draw_bone(img, coords[keypoints["left_elbow"]], coords[keypoints["left_wrist"]], thresh,  color=RED)
    
    img = draw_bone(img, coords[keypoints["right_shoulder"]], coords[keypoints["right_elbow"]], thresh,  color=GREEN)
    img = draw_bone(img, coords[keypoints["right_elbow"]], coords[keypoints["right_wrist"]], thresh,  color=GREEN)

    # legs
    img = draw_bone(img, coords[keypoints["left_hip"]], coords[keypoints["left_knee"]], thresh,  color=RED)
    img = draw_bone(img, coords[keypoints["left_knee"]],coords[keypoints["left_ankle"]], thresh,  color=RED)
    
    img = draw_bone(img, coords[keypoints["right_hip"]], coords[keypoints["right_knee"]], thresh,  color=GREEN)
    img = draw_bone(img, coords[keypoints["right_knee"]],coords[keypoints["right_ankle"]], thresh,  color=GREEN)

    # body
    img = draw_bone(img, coords[keypoints["left_shoulder"]], coords[keypoints["left_hip"]], thresh, color=WHITE)
    img = draw_bone(img, coords[keypoints["right_shoulder"]],  coords[keypoints["right_hip"]], thresh, color=WHITE)
    img = draw_bone(img, coords[keypoints["left_shoulder"]], coords[keypoints["right_shoulder"]], thresh, color=WHITE)
    img = draw_bone(img, coords[keypoints["left_hip"]], coords[keypoints["right_hip"]], thresh, color=WHITE)

    return img


def draw_bone(img, pt1, pt2, thresh, color):
    """
    Draws a line between two joints if their detection confidence is above the threshold,
    returns the plain image otherwise.

    Parameters
    ----------
    img : np.array (dtype: uint8)
        Input image.
    pt1 :  np.array (dtype: float)
        
        Coordinates and confidence score of the first point in format (x, y, p)
    pt2 : np.array (dtype: float)
        Coordinates and confidence score of the second point in format (x, y, p)
    thresh : float
        Detection confidence threshold.
    color : tuple (dtype: uin8) 
        Color tuple in BGR.

    Returns
    -------
    img : np.array (dtype: uint8)
        Output image.
    """
    if (pt1[-1] < thresh) or (pt2[-1] < thresh):
        return img
    
    H, W = img.shape[:2]
    Y1, X1 = (pt1[:2] * (H, W)).astype(int)
    Y2, X2 = (pt2[:2] * (H, W)).astype(int)

    img = cv2.line(img, (X1, Y1), (X2, Y2), color,  2)
    
    return img