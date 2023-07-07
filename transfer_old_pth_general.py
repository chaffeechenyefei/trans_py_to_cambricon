import torch
import os, argparse
pj = os.path.join

from config import config_param

if __name__ == "__main__":
    from checkpoint.checkpoint import CheckpointMgr
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_key')
    args = parser.parse_args()

    config_key = args.config_key
    if config_key not in config_param.keys():
        print('Error: Key {} not found'.format(config_key))
        exit(-1)
    else:
        print(config_param[config_key])

    print(config_param[config_key]['import'])
    exec(config_param[config_key]['import']) #import MLU_MODEL

    model_framework = config_key
    model_path = config_param[config_key]['path']
    model_date = config_param[config_key]['date']
    model_name = 'model_{}.pth'.format(model_date)
    new_model_name = 'model_{}_no_serial.pth'.format(model_date)#old format unzipped
    print('@@Model pth = {}'.format(model_name))


    model = MLU_MODEL()
    save_dir = model_path
    if hasattr(model, "load_checkpoint"):
        print('using class load_checkpoint method')
        model.load_checkpoint(pj(save_dir, model_name))
    else:
        print('using CheckpointMgr to load')
        checkpoint_op = CheckpointMgr(ckpt_dir=save_dir)
        checkpoint_op.load_checkpoint(model=model, ckpt_fpath=pj(save_dir, model_name), warm_load=False)

    torch.save(model.state_dict(),  pj(model_path, new_model_name),_use_new_zipfile_serialization=False)
