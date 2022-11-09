import cv2

RED = (0, 0, 204)
GREEN = (0, 204, 0)
WHITE =  (255, 255, 255)
BLUE = (204, 0, 0)

def draw_keypoints(img, coords, thresh, keypoints):
    """
    Draws the skeleton of a person. Keypoints below a given threshold are 
    ignored.

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
    image : np.array (dtype: uint8)
        Output image. 
    """
    image =  img.copy()
    # head
    if keypoints.get("left_eye", False):
        image = draw_bone(
            image, 
            coords[keypoints["left_eye"]], 
            coords[keypoints["left_ear"]], 
            thresh,  
            color=RED
            )    
        image = draw_bone(
            image, 
            coords[keypoints["left_eye"]], 
            coords[keypoints["nose"]], 
            thresh,  
            color=RED
            )

        image = draw_bone(
            image, 
            coords[keypoints["right_eye"]], 
            coords[keypoints["right_ear"]], 
            thresh,  
            color=GREEN
            )
        image = draw_bone(
            image, 
            coords[keypoints["right_eye"]], 
            coords[keypoints["nose"]], 
            thresh,  
            color=GREEN
            )
    elif keypoints.get("head_top", False):
        image = draw_bone(
            image, 
            coords[keypoints["head_top"]], 
            coords[keypoints["upper_neck"]], 
            thresh,  
            color=WHITE
            )

    # arms
    image = draw_bone(
        image, 
        coords[keypoints["left_shoulder"]], 
        coords[keypoints["left_elbow"]], 
        thresh,  
        color=RED
        )
    image = draw_bone(
        image, 
        coords[keypoints["left_elbow"]], 
        coords[keypoints["left_wrist"]], 
        thresh,  
        color=RED
        )
    
    image = draw_bone(
        image, 
        coords[keypoints["right_shoulder"]], 
        coords[keypoints["right_elbow"]], 
        thresh,  
        color=GREEN
        )
    image = draw_bone(
        image, 
        coords[keypoints["right_elbow"]], 
        coords[keypoints["right_wrist"]], 
        thresh,  
        color=GREEN
        )

    # legs
    image = draw_bone(
        image, 
        coords[keypoints["left_hip"]], 
        coords[keypoints["left_knee"]], 
        thresh,  
        color=RED
        )
    image = draw_bone(
        image, 
        coords[keypoints["left_knee"]],
        coords[keypoints["left_ankle"]], 
        thresh,  
        color=RED
        )
    
    image = draw_bone(
        image, 
        coords[keypoints["right_hip"]], 
        coords[keypoints["right_knee"]], 
        thresh,  
        color=GREEN
        )
    image = draw_bone(
        image, 
        coords[keypoints["right_knee"]],
        coords[keypoints["right_ankle"]], 
        thresh,  
        color=GREEN
        )

    # body
    if keypoints.get('left_shoulder', False):
        image = draw_bone(
            image, 
            coords[keypoints["left_shoulder"]], 
            coords[keypoints["left_hip"]], 
            thresh, 
            color=WHITE
            )
        image = draw_bone(
            image, 
            coords[keypoints["right_shoulder"]],  
            coords[keypoints["right_hip"]], 
            thresh, 
            color=WHITE
            )
        image = draw_bone(
            image, 
            coords[keypoints["left_shoulder"]],
            coords[keypoints["right_shoulder"]], 
            thresh, 
            color=WHITE
            )
        image = draw_bone(
            image, 
            coords[keypoints["left_hip"]], 
            coords[keypoints["right_hip"]], 
            thresh, 
            color=WHITE
            )
    elif keypoints.get('pelvis', False):
        image = draw_bone(
            image, 
            coords[keypoints["pelvis"]], 
            coords[keypoints["thorax"]], 
            thresh, 
            color=WHITE
            )
        image = draw_bone(
            image, 
            coords[keypoints["upper_neck"]], 
            coords[keypoints["thorax"]], 
            thresh, 
            color=WHITE
            )

    # draw keypoints
    image = draw_point(image, coords[:, :2], color=WHITE, radius="small")

    return image


def draw_bone(img, pt1, pt2, thresh, color):
    """
    Draws a line between two joints if their detection confidence is above the 
    threshold, returns the plain image otherwise.

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
    X1, Y1 = (pt1[:2] * (W, H)).astype(int)
    X2, Y2 = (pt2[:2] * (W, H)).astype(int)

    img = cv2.line(img, (X1, Y1), (X2, Y2), color,  2)
    
    return img


def draw_point(img, coords, color=BLUE, radius="big"):
    """
        Draws points on the image.

        Params
        ------
        img : np.array (dtype: uint8)
            Input image.
        coords : np.array (dtype: float)
            Keypoints prediction coordinates and scores in format (y, x, p).
        color: tuple
            Color of the circle in BGR. Default ot BLUE.
        radius: str
            If big it is 1% of the greatest dimension otherwise if 0.3%.

        Returns
        -------
        img : np.array (dtype: uint8)
            Output image. 
    """
    H, W = img.shape[:2]
    radius =  int(.01 * max(W, H)) if radius=="big" else int(.003 * max(W, H))
    
    if not coords.shape[1]:
        return img
    
    for coord in coords:
        X, Y = (coord * (W, H)).astype(int)
        img = cv2.circle(img, (X, Y), radius, color,  2)
    
    return img
