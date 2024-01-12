from models.model_single import Recognizer3D
from datasets.dataset import PoseC3DDataset
from datasets.preprocessing_pose_dataset import PreprocessingPoseFrames
import argparse
import yaml
import os
import numpy as np
import onnx
import onnxruntime

from collections import OrderedDict
import pickle

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import gc

gc.collect()
# import torch
torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description='Train Action Recognition Model')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--weight', help='config file path')

    args = parser.parse_args()

    return args


def load_model_recog(weights_path, device, args):
    output_device = device[0] if type(device) is list else device
    # model = Model(num_class=6, num_point=17, num_person=2, graph="graph.ntu_rgb_d.Graph", graph_args=gr_arg, in_channels=2,
    #  drop_out=0, adaptive=True)
    model = Recognizer3D(backbone=args['model']['backbone']['args'],
                       cls_head=args['model']['cls_head']['args'])
    print(model)

    if weights_path:
        if '.pkl' in weights_path:
            with open(weights_path, 'r') as f:
                w = pickle.load(f)
        else:
            w = torch.load(weights_path)

        w = OrderedDict([[k.split('module.')[-1], v] for k, v in w.items()])  # .cuda(output_device)
        # w = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in w.items()])

        try:
            model.load_state_dict(w)
        except:
            state = model.state_dict()
            diff = list(set(state.keys()).difference(set(w.keys())))
            print('Can not find these weights:')
            for d in diff:
                print('  ' + d)
            state.update(w)
            model.load_state_dict(state)

    return model


if __name__ == '__main__':
    args = parse_args()
    config = args.config
    f = open(config, 'r')
    default_arg = yaml.load(f)
    weight_path = args.weight
    dst_path = weight_path.replace('.pt', '.onnx')
    device = torch.device("cpu")
    model = load_model_recog(weights_path=weight_path, device=device, args=default_arg)
    model.eval()
    model.to(device=device)
    test_input = torch.randn(1, 17, 48, 64, 64).to(device=device)

    a = model(test_input)
    print(a)
    torch.onnx.export(model, test_input, dst_path, export_params=True, verbose=False, opset_version=11)
    # pipeline = PreprocessingPoseFrames(pipeline=default_arg['val_pipeline'],test_mode=True)
    np_input = test_input.numpy()
    # print(np_input)
    oss_sess = onnxruntime.InferenceSession(dst_path)
    input_tensor = dict()
    input_tensor['x.1'] = np_input
    output = oss_sess.run(None, input_tensor)

    print(output)

