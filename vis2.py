import cv2
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.body21 import Body21TrainDataset
from datasets.transformations import ConvertKeypoints2, Scale, Rotate, CropPad, Flip2
from modules.keypoints_2 import BODY_PARTS_PAF_IDS

kp_names = ['Nose','Neck',
            'RShoulder','RElbow','RWrist','LShoulder','LElbow','LWrist',
            'MidHip','RHip','RKnee','RAnkle','LHip','LKnee', 'LAnkle',
            'REye','LEye','REar','LEar',
            'LBigToe','RBigToe',
            'Background']
pf_names = ['Spine','RShoulder','LShoulder','RArm','RForeArm','LArm',
            'LForeArm','RHip','RThigh','RCalf','LHip','LThigh','LCalf',
            'Neck','REye','ERar','LEye','LEar', 'RShoulderEar','LShoulderEar',
            'LFoot','RFoot']

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader

def vis_dataset(prepared_train_labels, train_images_folder):
    stride = 8
    sigma = 7
    path_thickness = 1
    dataset = Body21TrainDataset(prepared_train_labels, train_images_folder,
                                 stride, sigma, path_thickness,
                                 transform=transforms.Compose([
                                    ConvertKeypoints2(),
                                    Scale(),
                                    Rotate(pad=(128, 128, 128)),
                                    CropPad(pad=(128, 128, 128)),
                                    Flip2()]))
    train_loader = DataLoader(dataset)
    for batch_data in train_loader:
        images = batch_data['image']
        keypoint_masks = batch_data['keypoint_mask']
        paf_masks = batch_data['paf_mask']
        keypoint_maps = batch_data['keypoint_maps']
        paf_maps = batch_data['paf_maps']
        
        #num_kepoints = batch_data['label']['num_keypoints'][0].item()

        # if num_kepoints < 21:
        #     continue

        image_bgr = np.uint8(images[0].numpy().transpose(1, 2, 0) * 256 + 128)
        cv2.imshow('image', image_bgr)
        cv2.moveWindow('image', 384, 0)

        for i in range(len(keypoint_maps[0])):
            heat_map = cv2.resize(keypoint_maps[0][i].numpy(), (images[0].shape[2], images[0].shape[1]))
            heat_map = np.expand_dims(np.uint8(heat_map * 255), -1)
            debug_map = image_bgr // 2 + heat_map // 2
            win_name = 'keypoint: {}'.format(kp_names[i])
            cv2.imshow(win_name, debug_map)
            cv2.moveWindow(win_name, 0, 0)
            cv2.waitKey(0)
            cv2.destroyWindow(win_name)

        for i in range(len(BODY_PARTS_PAF_IDS)):
            pair = BODY_PARTS_PAF_IDS[i]
            paf_map_x = np.abs(paf_maps[0][pair[0]].numpy())
            paf_map_y = np.abs(paf_maps[0][pair[1]].numpy())
            paf_map = np.fmin(np.fmax(paf_map_x, paf_map_y)*100, 1.0)
            paf_map = cv2.resize(paf_map, (images[0].shape[2], images[0].shape[1]))
            paf_map = np.expand_dims(np.uint8(abs(paf_map) * 255), -1)
            debug_map = image_bgr // 2 + paf_map // 2
            win_name = 'part: {}'.format(pf_names[i])
            cv2.imshow(win_name, debug_map)
            cv2.moveWindow(win_name, 0, 0)
            cv2.waitKey(0)
            cv2.destroyWindow(win_name)

if __name__ == '__main__':
    vis_dataset('data/prepared_train_annotation_1.pkl', 'coco/train2017')