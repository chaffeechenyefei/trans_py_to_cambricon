from models.unet import UNetResNet18
# from models.unet import UNetResNet18_Shift
import torch

B,T,C,H,W = 5,2,3,224,224
with torch.no_grad():
    model = UNetResNet18(n_classes=1,n_segment=T)
    x = torch.rand(B,C,T,H,W)
    y = model(x)

    # model = UNetResNet18_Shift(n_classes=2,n_segment=T)
    # x = torch.rand(B, C, T,  H, W)
    # y = model(x)

    print('y =', y.shape)



