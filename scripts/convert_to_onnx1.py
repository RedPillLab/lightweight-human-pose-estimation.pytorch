import sys, os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.with_mobilenet import PoseEstimationWithMobileNet, PoseEstimationWithMobileNet1
from modules.load_state import load_state

def convert_to_onnx(net, output_name):
    #input = torch.randn(1, 3, 288, 512)
    input = torch.randn(1, 3, 240, 320)
    #input_names = ['data']
    #output_names = ['stage_0_output_1_heatmaps', 'stage_0_output_0_pafs',
    #                'stage_1_output_1_heatmaps', 'stage_1_output_0_pafs']
    input_names = ['image']
    output_names = ['net_output']
    torch.onnx.export(net, input, output_name, verbose=True,
                      input_names=input_names, output_names=output_names)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    # parser.add_argument('--output-name', type=str, default='human-pose-estimation.onnx',
    #                     help='name of output model in ONNX format')
    # args = parser.parse_args()

    #net = PoseEstimationWithMobileNet(num_heatmaps=26, num_pafs=52, num_refinement_stages=1)
    net = PoseEstimationWithMobileNet1(num_heatmaps=26, num_pafs=52)
    checkpoint = torch.load('body25_checkpoints/checkpoint_iter_465000.pth')
    load_state(net, checkpoint)

    convert_to_onnx(net, 'body25_checkpoints/pose2d_mobile_320x240.onnx')
