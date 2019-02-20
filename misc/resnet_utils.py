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
        # fc_ori = self.resnet.avgpool(x).view(x.size(0), -1).squeeze()
        # print(fc.shape, fc_ori.shape)
        # print(fc[1234:1254])
        # print(fc_ori[1234:1254])
        # print(torch.equal(fc, fc_ori))

        att = F.adaptive_avg_pool2d(
                x,
                [att_size,att_size]
                ).squeeze().permute(1, 2, 0) # 14, 14, 2048
        # print(att.shape)
        
        if not visualConceptReturn:
            return fc, att
        else:
            # visual concept
            # print(vc.shape)
            vc = self.resnet.fc(fc) # 4267
            return vc, att
            return fc, vc

