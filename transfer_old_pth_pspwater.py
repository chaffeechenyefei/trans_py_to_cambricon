import torch
import os, argparse
pj = os.path.join


if __name__ == "__main__":
    # from models.unet import UNet2D
    from psp_models import *
    from checkpoint.checkpoint import CheckpointMgr
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_date',default='20211119')
    args = parser.parse_args()

    model_framework = 'pspwater'
    model_path = 'pth/pspwater/milestone'
    model_date = args.model_date
    model_name = 'model_{}.pth'.format(model_date)
    new_model_name = 'model_{}_no_serial.pth'.format(args.model_date)#old format unzipped
    print('@@Model pth = {}'.format(model_name))

    psp_cfg = dict(
        backbone=dict(
            depth=18,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            deep_stem=True,
            avg_down=False,
            norm_eval=False,
            contract_dilation=True),
        decode_head=dict(
            in_channels=512,
            in_index=3,
            channels=128,
            pool_scales=(1, 2, 4, 8),
            dropout_ratio=0.1,
            num_classes=2,
            align_corners=False),
        auxiliary_head=dict(
            in_channels=256,
            in_index=2,
            channels=64,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=2,
            align_corners=False))
    model = PspNet(psp_cfg)
    save_dir = model_path
    checkpoint_op = CheckpointMgr(ckpt_dir=save_dir)
    checkpoint_op.load_checkpoint(model=model, ckpt_fpath=pj(save_dir, model_name), warm_load=False)

    torch.save(model.state_dict(),  pj(model_path, new_model_name),_use_new_zipfile_serialization=False)
