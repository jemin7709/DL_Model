import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN16(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()

        feats = list(models.vgg16(pretrained=True).features.children())
        self.feats = nn.Sequential(*feats[0:17])
        self.feat4 = nn.Sequential(*feats[17:24] )
        self.feat5 = nn.Sequential(*feats[24:31])
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout() ,
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score_fconn = nn.Conv2d(4096, num_classes, 1)
        self.score_feat4 = nn.Conv2d( 512, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        feat4 = self.feat4(feats)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
        score += score_feat4

        return F.upsample_bilinear(score, x.size()[2:])