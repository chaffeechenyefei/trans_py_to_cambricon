import torch
import os, argparse
pj = os.path.join


if __name__ == "__main__":
    # from models.unet import UNet2D
    from smoking_models.models import SmokingModel_1_MLU
    from checkpoint.checkpoint import CheckpointMgr
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_date',default='20211119')
    args = parser.parse_args()

    model_framework = 'smoking-r34'
    model_path = 'pth/smoking-r34'
    model_date = args.model_date
    model_name = 'model_{}.pth'.format(model_date)
    new_model_name = 'model_{}_no_serial.pth'.format(args.model_date)#old format unzipped
    print('@@Model pth = {}'.format(model_name))


    model = SmokingModel_1_MLU()
    save_dir = model_path
    checkpoint_op = CheckpointMgr(ckpt_dir=save_dir)
    checkpoint_op.load_checkpoint(model=model, ckpt_fpath=pj(save_dir, model_name), warm_load=False)

    torch.save(model.state_dict(),  pj(model_path, new_model_name),_use_new_zipfile_serialization=False)
