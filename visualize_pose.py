import os
import cv2
import json
import numpy as np
PART_NAMES = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
              "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "Background"]
PART_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
              [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]
PAIR_COLORS = [[255,     0,    85],
[255,     0,     0],
[255,    85,     0],
[255,   170,     0],
[255,   255,     0],
[170,   255,     0],
[ 85,   255,     0],
[  0,   255,     0],
[  0,   255,    85],
[  0,   255,   170],
[  0,   255,   255],
[  0,   170,   255],
[  0,    85,   255],
[  0,     0,   255],
[255,     0,   170],
[170,     0,   255],
[255,     0,   255],
[ 85,     0,   255]]

def vis_pose(img, joints):
    to_show = img.copy()
    for body in joints:
        joint = np.asarray(body['joints'], dtype=np.int)
        for i, pair in enumerate(PART_PAIRS):
            cv2.line(to_show, tuple(joint[pair[0]][:2]), tuple(joint[pair[1]][:2]), color=PAIR_COLORS[i][::-1], thickness=3)
    return to_show

def demo_pose(vid_path, pose_dir):
    vid = cv2.VideoCapture(vid_path)

    frame_ind = 0
    while True:
        ret, frame = vid.read()
        if not ret or frame is None:
            break
        with open(os.path.join(pose_dir, '{:07d}.json'.format(frame_ind)), 'r') as f:
            pose = json.load(f)

        to_show = vis_pose(frame, pose['bodies'])
        cv2.imshow('img', to_show)
        cv2.waitKey(30)
        frame_ind += 1
    return 0

if __name__ == '__main__':
    demo_pose('data/video/M_00001.avi', 'data/pose/001/M_00001')
