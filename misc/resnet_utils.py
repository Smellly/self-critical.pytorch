import torch
import torch.nn as nn
import torch.nn.functional as F

class myResnet(nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=14, visualConceptReturn=False):
        # print('vc flag:', visualConceptReturn)
        x = img.unsqueeze(0)
        # print(x.shape)

        x = self.resnet.conv1(x)   # 64   * 112 * 112
        # print(x.shape)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) # 64   * 56 * 56
        # print(x.shape)

        x = self.resnet.layer1(x)  # 256  * 56 * 56
        # print(x.shape)
        x = self.resnet.layer2(x)  # 512  * 28 * 28
        # print(x.shape)
        x = self.resnet.layer3(x)  # 1024 * 14 * 14
        # print(x.shape)
        x = self.resnet.layer4(x)  # 2048 * 7  * 7 
        # print(x.shape)

        fc = x.mean(3).mean(2).squeeze() # 2048
        # print(fc.shape)

        # visual concept
        vc = self.resnet.fc(fc) # 4267
        # print(vc.shape)

        att = F.adaptive_avg_pool2d(
                x,
                [att_size,att_size]
                ).squeeze().permute(1, 2, 0) # 14, 14, 2048
        # print(att.shape)
        
        if not visualConceptReturn:
            return fc, att
        else:
            return fc, vc

