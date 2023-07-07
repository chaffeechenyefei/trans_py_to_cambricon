import torch
import os, argparse
pj = os.path.join


if __name__ == "__main__":
    from models.unet import UNet2D
    from checkpoint.checkpoint import CheckpointMgr
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_key',default='293')
    args = parser.parse_args()

    model_framework = 'unetwater'
    model_path = 'pth/UNet2D_Water/milestone'
    model_key = args.model_key
    model_name = 'model_{}.pth'.format(model_key)
    new_model_name = 'model_{}_no_serial.pth'.format(args.model_key)#old format unzipped
    print('@@Model pth = {}'.format(model_name))

    model = UNet2D(3,1)
    save_dir = model_path
    checkpoint_op = CheckpointMgr(ckpt_dir=save_dir)
    checkpoint_op.load_checkpoint(model=model, ckpt_fpath=pj(save_dir, model_name), warm_load=False)

    torch.save(model.state_dict(),  pj(model_path, new_model_name),_use_new_zipfile_serialization=False)
