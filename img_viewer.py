from imutils import paths
import numpy as np
import argparse
import imutils
import sys
import cv2
import os

from bodypose.demo.graphics import draw_keypoints, draw_point
from config import COCO_KEYPOINT_DICT, MPII_KEYPOINT_DICT

green = (0, 255, 0)
red =  (0, 0, 255)
blue = (255, 0, 0)

def main(imgPaths, labelPaths, idx, kpts_dict):
    i = idx

    while i < len(imgPaths):

        # select the img and corresponding label file
        imgPath = imgPaths[i]
        imgName = imgPath.split(os.sep)[-1]
        
        kptsTxtPath = ".".join(imgName.split(".")[:-1]) + "_kpts.txt"
        kptsTxtPath = os.sep.join([labelPaths, kptsTxtPath])

        cntrsTxtPath = ".".join(imgName.split(".")[:-1]) + "_cntrs.txt"
        cntrsTxtPath = os.sep.join([labelPaths, cntrsTxtPath])

        # get keypoints coordinates
        c_kpts = np.nan_to_num(np.loadtxt(kptsTxtPath))
        c_cntrs = np.nan_to_num(np.loadtxt(cntrsTxtPath))
        if len(c_cntrs.shape)==1:
            c_cntrs = np.expand_dims(c_cntrs, axis=0)

        # read the image and draw the bbox
        img = cv2.imread(imgPath)
        H, W = img.shape[:2]

        img = draw_keypoints(img, c_kpts, -1, kpts_dict)
        img = draw_point(img, c_cntrs)

        # reshape according to its size
        if H > W:
            img = imutils.resize(img, height=800)
        else:
            img = imutils.resize(img, width=1200)

        print("[INFO] image {}/{}".format(i, len(imgPaths)))

        # display the image
        filename = imgPath.split(os.sep)[-1]
        cv2.namedWindow(filename)
        cv2.moveWindow(filename, 20,20)
        cv2.imshow(filename, img)
        key = cv2.waitKey(0)

        # if j is pressed go to the previous image
        if key & 0xFF == ord("j"):
            i -= 1
            cv2.destroyAllWindows()

        # if l is pressed go to the nex image
        elif key & 0xFF == ord("l"):
            i +=1
            cv2.destroyAllWindows()

        # if q is pressed exit
        elif key & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            sys.exit()

        # if s is pressed save the image with bboxes
        elif key & 0xFF == ord("s"):
            outImgName = ".".join(filename.split(".")[:-1]) + "_out.png"
            cv2.imwrite(outImgName, img)
            print(f"[INFO] Saved imgage with bboxes @ {outImgName}.")

        # if s is pressed save the image with bboxes
        elif key & 0xFF == ord("d"):
            os.remove(imgPath)
            os.remove(kptsTxtPath)
            os.remove(cntrsTxtPath)
            i+=1
            print(f"[INFO] Removed image  and labels.")
            cv2.destroyAllWindows()
    
    cv2.destroyAllWindows()
    sys.exit()


if __name__ == "__main__":
    
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image",
        help="path to image")
    ap.add_argument("-d", "--directory",
        help="path to directory")
    ap.add_argument("-l", "--label_dir",
        help="path to label directory")
    ap.add_argument("--idx", type=int, default=0,
        help="starting index.")
    ap.add_argument("--ds", type=str, default='coco',
        help="Dataset name")
    args = vars(ap.parse_args())

    # accept either a single image or an entire directory
    if args["image"] is not None:
        imgPaths = [args["image"]]
    elif args["directory"] is not None:
        imgPaths = list(paths.list_images(args["directory"]))
    else:
        print("[INFO] you must supply a path either to image or to directory")

    # path to label folder -  default: imgPath
    if args["label_dir"] is None:
        labelPaths = os.sep.join(imgPaths[0].split(os.sep)[:-1])
    else:
        labelPaths = args["label_dir"]

    if args['ds']=='coco':
        kpts_dict = COCO_KEYPOINT_DICT
    elif args['ds']=='mpii':
        kpts_dict = MPII_KEYPOINT_DICT

    main(imgPaths, labelPaths, args["idx"], kpts_dict)