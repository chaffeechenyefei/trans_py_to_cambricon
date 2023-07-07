import torch
import os, argparse
pj = os.path.join


if __name__ == "__main__":
    from models.unet import UNetResNet18_BN
    from checkpoint.checkpoint import CheckpointMgr
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_key',default='199')
    parser.add_argument('--t',default=2)
    args = parser.parse_args()

    T = args.t
    model_framework = 'unetResNet18_bn'
    model_path = 'pth/UNetResNet18_BN/bs2/milestone'
    model_key = args.model_key
    model_name = 'model_{}.pth'.format(model_key)
    new_model_name = 'model_{}_no_serial.pth'.format(args.model_key)#old format unzipped
    print('@@Model pth = {}'.format(model_name))

    model = UNetResNet18_BN(n_classes=1,n_segment=2, use_bn=False)
    save_dir = model_path
    checkpoint_op = CheckpointMgr(ckpt_dir=save_dir)
    checkpoint_op.load_checkpoint(model=model, ckpt_fpath=pj(save_dir, model_name), warm_load=False)

    torch.save(model.state_dict(),  pj(model_path, new_model_name),_use_new_zipfile_serialization=False)
