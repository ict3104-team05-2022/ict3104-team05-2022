import numpy as np
import os

from omegaconf import OmegaConf
from tqdm import tqdm

from utils2.utils import build_cfg_path, form_list_from_user_input, sanity_check

import os
import os.path


def main(args_cli):
    # config
    args_yml = OmegaConf.load(build_cfg_path(args_cli.feature_type))
    # the latter arguments are prioritized
    args = OmegaConf.merge(args_yml, args_cli)
    # OmegaConf.set_readonly(args, True)
    sanity_check(args)

    # verbosing with the print -- haha (TODO: logging)
    print(OmegaConf.to_yaml(args))
    if args.on_extraction in ['save_numpy', 'save_pickle']:
        # print(f'Saving features to {args.output_path}')
        print(f'Saving features to ./data/dataset/v_iashin_i3d/')
    print('Device:', args.device)

    # import are done here to avoid import errors (we have two conda environements)
    if args.feature_type == 'i3d':
        from models2.i3d.extract_i3d import ExtractI3D as Extractor
    elif args.feature_type == 'r21d':
        from models2.r21d.extract_r21d import ExtractR21D as Extractor
    elif args.feature_type == 's3d':
        from models2.s3d.extract_s3d import ExtractS3D as Extractor
    elif args.feature_type == 'vggish':
        from models2.vggish.extract_vggish import ExtractVGGish as Extractor
    elif args.feature_type == 'resnet':
        from models2.resnet.extract_resnet import ExtractResNet as Extractor
    elif args.feature_type == 'raft':
        from models2.raft.extract_raft import ExtractRAFT as Extractor
    elif args.feature_type == 'pwc':
        from models2.pwc.extract_pwc import ExtractPWC as Extractor
    elif args.feature_type == 'clip':
        from models2.clip.extract_clip import ExtractCLIP as Extractor
    else:
        raise NotImplementedError(
            f'Extractor {args.feature_type} is not implemented.')

    extractor = Extractor(args)

    # unifies whatever a user specified as paths into a list of paths
    video_paths = form_list_from_user_input(
        args.video_paths, args.file_with_video_paths, to_shuffle=True)

    print(f'The number of specified videos: {len(video_paths)}')

    for video_path in tqdm(video_paths):
        # Get the video file name from the video path
        video_file_name = video_path.split('/')[-1]
        video_file_name = video_file_name.split('.')[0]

        feature_dict = extractor.extract(video_path)
        rgb = list(feature_dict.items())[0]

        # Reshape the data to fit into the TSU model
        rgb_data = np.expand_dims(rgb[1], axis=(2, 1))
        rgb_dir = '../data/dataset/v_iashin_i3d/'
        rgb_dir_exists = os.path.isdir(rgb_dir)

        if not rgb_dir_exists:
            os.mkdir(rgb_dir)

        np.save(rgb_dir + video_file_name + ".npy", rgb_data)


if __name__ == '__main__':
    args_cli = OmegaConf.from_cli()
    main(args_cli)
