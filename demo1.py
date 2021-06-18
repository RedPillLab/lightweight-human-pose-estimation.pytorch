import argparse

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints_1 import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose_1 import Pose, track_poses
from val import normalize, pad_width
from modules.keypoints_1 import BODY_PARTS_PAF_IDS

kp_names = ['Nose','Neck',
            'RShoulder','RElbow','RWrist','LShoulder','LElbow','LWrist',
            'MidHip','RHip','RKnee','RAnkle','LHip','LKnee', 'LAnkle',
            'REye','LEye','REar','LEar',
            'LBigToe','LSmallToe','LHeel','RBigToe','RSmallToe','RHeel',
            'Background']
pf_names = ['Spine','RShoulder','LShoulder','RArm','RForeArm','LArm',
            'LForeArm','RHip','RThigh','RCalf','LHip','LThigh','LCalf',
            'Neck','REye','ERar','LEye','LEar', 'RShoulderEar','LShoulderEar',
            'LFoot','LToe','LHeel','RFoot','RToe','RHeel']

class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio,
                          interpolation=cv2.INTER_CUBIC)

    # for i in range(heatmaps.shape[-1]):
    #     heatmap = heatmaps[:,:,i]
    #     heat_map = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    #     heat_map = np.fmin(np.fmax(heat_map, 0.0), 1.0)
    #     heat_map = np.expand_dims(np.uint8(heat_map * 255), -1)
    #     debug_map = img // 2 + heat_map // 2
    #     win_name = 'keypoint: {}'.format(kp_names[i])
    #     cv2.imshow(win_name, debug_map)
    #     cv2.moveWindow(win_name, 0, 0)
    #     cv2.waitKey(0)
    #     cv2.destroyWindow(win_name)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    # for i in range(pafs.shape[-1] // 2):
    #     paf = np.zeros((pafs.shape[0], pafs.shape[1], 3), dtype=np.float32)
    #     pair = BODY_PARTS_PAF_IDS[i]
    #     paf[:,:,2] = np.abs(pafs[:,:,pair[0]])
    #     paf[:,:,1] = np.abs(pafs[:,:,pair[1]])
    #     paf = np.fmin(np.fmax(paf, -1.0), 1.0)
    #     paf = cv2.resize(paf, (img.shape[1], img.shape[0]))
    #     paf = np.uint8(abs(paf) * 255)
    #     debug_map = img // 2 + paf // 2
    #     win_name = 'part: {}'.format(pf_names[i])
    #     cv2.imshow(win_name, debug_map)
    #     cv2.moveWindow(win_name, 0, 0)
    #     cv2.waitKey(0)
    #     cv2.destroyWindow(win_name)

    return heatmaps, pafs, scale, pad

def run_demo(net, image_provider, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    for img in image_provider:
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 26th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs,
                                                      pose_entry_size=27, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][25])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 33:
                delay = 0
            else:
                delay = 1

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description='''Lightweight human pose estimation python demo.
    #                    This is just for quick results preview.
    #                    Please, consider c++ demo for the best performance.''')
    # parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    # parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    # args = parser.parse_args()

    # if args.video == '':
    #     raise ValueError('--video has to be provided')

    # net = PoseEstimationWithMobileNet(num_heatmaps=26, num_pafs=52)
    # checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    # load_state(net, checkpoint)

    # frame_provider = VideoReader(args.video)

    # run_demo(net, frame_provider, 256, False, True, True)

    net = PoseEstimationWithMobileNet(num_heatmaps=26, num_pafs=52, num_refinement_stages=1)
    checkpoint = torch.load('body25_checkpoints/checkpoint_iter_465000.pth', map_location='cpu')
    load_state(net, checkpoint)

    frame_provider = VideoReader('D:/projects/MotioNet/video/beyonce.mp4')

    run_demo(net, frame_provider, 256, False, True, True)