# Environment
```commandline
export http_tsinghua='https://pypi.tuna.tsinghua.edu.cn/simple'
pip install pretrainedmodels imgaug scikit-learn tqdm opencv-python pillow -i $http_tsinghua
pip install --upgrade scikit-image
pip install easydict prefetch_generator -i $http_tsinghua
```

# Usage
* 登录服务器
    ```commandline
        ~/.ssh/config
        host mlu270
          hostname 192.168.184.156
          user root
          IdentityFile ~/.ssh/id_rsa
    ```

* 进入容器并切换到pytorch的环境下
    ```commandline
    runmlupytorch
    sourcepy3torch
    ```

* 通过transfer_old_pth_general.py在一台主机上生成没有序列化压缩过的pth文件。
* 通过mlu_general.py, 在寒武纪270机器上生成int8量化后的在线模型
* 通过genoff_general.py, 分别生成270/220上可以运行的离线模型

## 1.打电话分类(单输入通用接口)
```commandline
(*) python transfer_old_pth_general.py --config_key phoning-r34
python mlu_general.py --config_key phoning-r34 --mlu 0 --quantization 1 --half_input 1 --data ./data/images/ --batch_size 2
python genoff_general.py -config_key phoning-r34 -fake_device 0 -input_format 0 -half_input 1 -core_number 1
python genoff_general.py -config_key phoning-r34 -fake_device 1 -mcore MLU220 -input_format 0 -half_input 1 -core_number 1
```

## 2.骨架检测(单输入通用接口) yutao posenet-r18

```commandline
(*) python transfer_old_pth_general.py --config_key posenet-r18
python mlu_general.py --config_key posenet-r18 --mlu 0 --quantization 1 --half_input 1 --data ./data/images/ --batch_size 2
python genoff_general.py -config_key posenet-r18 -fake_device 0 -input_format 0 -half_input 1 -core_number 1
python genoff_general.py -config_key posenet-r18 -fake_device 1 -mcore MLU220 -input_format 0 -half_input 1 -core_number 1

```

## 3.人脸属性识别(单输入通用接口) shawn efficientnet

```commandline
(*) python transfer_old_pth_general.py --config_key faceattr-effnet
python mlu_general.py --config_key faceattr-effnet --mlu 0 --quantization 1 --half_input 1 --data ./data/images/ --batch_size 2
python genoff_general.py -config_key faceattr-effnet -fake_device 0 -input_format 0 -half_input 1 -core_number 1
python genoff_general.py -config_key faceattr-effnet -fake_device 1 -mcore MLU220 -input_format 0 -half_input 1 -core_number 1
```

(*) ```python transfer_old_pth_general.py``` 需要在其它机器上运行，生成没有序列化压缩过的pth文件
