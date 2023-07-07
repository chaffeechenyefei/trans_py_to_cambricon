import torch
import os, argparse
pj = os.path.join


if __name__ == "__main__":
    from checkpoint.checkpoint import CheckpointMgr
    from headpose_models.head_pose_inference_mlu import HopenetMBV2
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_date',default='20211223')
    args = parser.parse_args()

    model_framework = 'headpose'
    model_path = 'pth/headpose'
    model_date = args.model_date
    model_name = 'model_{}.pth'.format(model_date)
    new_model_name = 'model_{}_no_serial.pth'.format(args.model_date)#old format unzipped
    print('@@Model pth = {}'.format(model_name))

    model = HopenetMBV2()
    save_dir = model_path
    checkpoint_op = CheckpointMgr(ckpt_dir=save_dir)
    checkpoint_op.load_checkpoint(model=model, ckpt_fpath=pj(save_dir, model_name), warm_load=False)

    torch.save(model.state_dict(),  pj(model_path, new_model_name),_use_new_zipfile_serialization=False)
