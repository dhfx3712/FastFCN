
数据
VOCdevkit/VOC2007



模型
backone resnet50 troch.vision通用
from torchvision import models
import torch
model = models.resnet50(pretrained=False)
model.load_state_dict(torch.load('/Users/admin/Downloads/models/resnet50-0676ba61.pth'), strict=False)


fcn 
encoding/models/fcn.py
get_fcn -->FCN()-->self.base_forward(x)
        encoding/models/base.py
        self.pretrained = resnet.resnet50() backbone
        self.pretrained.layer1() resnet结构的四层输出
        self.jpu = JPU()
            encoding/nn/customize.py
            nn.Conv2d(dilation) 空洞卷积
            feat = torch.cat(feats, dim=1)按第一层大小调整二、三的尺寸，然后cat起来
            以feat为基础，做4次空洞卷积，然后在cat起来
            
FCN()-->self.head(c4)-->FCNHead(2048, nclass, norm_layer)
    interpolate插值采样
    

    


损失

if not self.se_loss and not self.aux:
super(SegmentationLosses, self).forward(*inputs)
input和target均是列表
F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
                               
                               
                               





encnet结构（一个分割损失一个分类损失）
python experiments/segmentation/train.py --dataset pascal_voc  --jpu JPU --no-cuda --se-loss --lateral
模型
encoding/models/encnet.py
features = self.base_forward(x)
x = list(self.head(*features)) self.head 如何处理
    self.head = EncHead() -->(self.connect,self.fusion,self.encmodule)



self.encmodule
    self.encoding(x) embedding处理
        encoding/nn/encoding.py
        encoding.nn.Encoding()-->F.softmax(dim=2)第三维归一化操作-->aggregate??
        torch.nn.BatchNorm1d(ncodes) encoding的输出维度要与batchnorm匹配上

F.softmax dim
https://blog.csdn.net/qq_43359515/article/details/126083252
            
            

--lateral
EncHead_input : torch.Size([1, 256, 120, 120]),torch.Size([1, 512, 60, 60]),torch.Size([1, 1024, 30, 30]),torch.Size([1, 2048, 60, 60]),feat_shape,torch.Size([1, 512, 60, 60])
c2 = self.connect[0](inputs[1])
c3 = self.connect[1](inputs[2])
print (f'c2,{c2.shape},c3,{c3.shape}')
feat = self.fusion(torch.cat([feat, c2, c3], 1))  
c2与c3维度不匹配，拼接报错

self.fusion = nn.Sequential(
        nn.Conv2d(3*512, 512, kernel_size=3, padding=1, bias=False),
        norm_layer(512),
        nn.ReLU(inplace=True))
fusion的输入维度是cat拼接后的维度



deeplabv3(两个分割损失)

python experiments/segmentation/train.py --dataset pascal_voc --model deeplab --jpu JPU --no-cuda --aux  --dilated

