import torch
import argparse
import numpy as np
import multiprocessing

from utils import *
from solver import Solver
from config import get_args, get_config
from data_loader import get_loader



def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        generator = torch.Generator(device='cuda')
        torch.cuda.manual_seed_all(seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        generator = torch.Generator()

    generator.manual_seed(seed)
    return generator


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    args = get_args()
    dataset = str.lower(args.dataset.strip())
    args.data_dir = '/data1'
    pose_dir = '/data1/features_mtcnn_head_pose_aggregated'
    audio_dir = '/data1'
    is09_dir = '/data1/IS09NPZ'

    generator = set_seed(args.seed)

    print("Start loading the data....")

    train_config = get_config(
        dataset,
        mode='train',
        batch_size=args.batch_size,
        visual_encoder_type=args.visual_encoder_type,
        acoustic_encoder_type=args.acoustic_encoder_type,
        use_transformer_fusion=getattr(args, 'use_transformer_fusion', False),
        save_name=args.save_name
    )

    valid_config = get_config(
        dataset,
        mode='valid',
        batch_size=args.batch_size,
        visual_encoder_type=args.visual_encoder_type,
        acoustic_encoder_type=args.acoustic_encoder_type,
        use_transformer_fusion=getattr(args, 'use_transformer_fusion', False),
        save_name=args.save_name
    )

    test_config = get_config(
        dataset,
        mode='test',
        batch_size=args.batch_size,
        visual_encoder_type=args.visual_encoder_type,
        acoustic_encoder_type=args.acoustic_encoder_type,
        use_transformer_fusion=getattr(args, 'use_transformer_fusion', False),
        save_name=args.save_name
    )

    train_loader = get_loader(args.data_dir, pose_dir, audio_dir, is09_dir, args.batch_size, phase='train', shuffle=True, generator=generator)
    print('Training data loaded!')

    valid_loader = get_loader(args.data_dir, pose_dir, audio_dir, is09_dir, args.batch_size, phase='valid', shuffle=False, generator=generator)
    print('Validation data loaded!')

    test_loader = get_loader(args.data_dir, pose_dir, audio_dir, is09_dir, args.batch_size, phase='test', shuffle=False, generator=generator)
    print('Test data loaded!')
    print('Finish loading the data....')

    torch.autograd.set_detect_anomaly(True)

    args.n_class = 3
    args.d_vin = 512
    args.d_pose = 272
    args.d_aud = 768
    args.d_audio_in = 768
    args.d_audio_hidden = 128
    args.d_is09 = 384
    args.dataset = dataset

    args.when = args.when
    args.criterion = 'CrossEntropyLoss'

    solver = Solver(
        args,
        train_loader=train_loader,
        dev_loader=valid_loader,
        test_loader=test_loader,
        is_train=True,
        save_name=args.save_name
    )

    if torch.cuda.is_available():
        solver.model.to('cuda')

    solver.train_and_eval()





