import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, r = 16):
        super(SEBlock, self).__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels//r),
            nn.ReLU(),
            nn.Linear(in_features=in_channels//r, out_features=in_channels)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        global_pool = self.global_pool(x)
        global_pool = global_pool.reshape(b, c)
        
        fc = self.fc(global_pool)
        fc = fc.reshape(b, c, 1, 1)
        
        sigmoid = self.sigmoid(fc)
        
        return x * sigmoid

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_ch, out_ch, down=False):
        super(BottleNeck, self).__init__()
        self.in_ch = in_ch
        self.down = down
        if self.down == True:
            self.B_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(num_features=out_ch),
                nn.ReLU()
            )
        elif self.down == False:
            self.B_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_ch * self.expansion, out_channels=out_ch, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(num_features=out_ch),
                nn.ReLU()
            )
        
        self.B_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU()
        )
        
        self.B_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch * self.expansion, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_ch * self.expansion),
            nn.ReLU()
        )
        
        self.seblock = SEBlock(in_channels=out_ch * self.expansion)
        
        if self.down == True:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch * self.expansion, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(num_features=out_ch * self.expansion)
            )
        elif self.down == False:
            self.downsample = None
            
        self.equal = nn.Conv2d(in_channels=in_ch * 4, out_channels=out_ch * self.expansion, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        downsample = x
        B_conv1 = self.B_conv1(x)
        B_conv2 = self.B_conv2(B_conv1)
        B_conv3 = self.B_conv3(B_conv2)
        
        if self.downsample != None:
            downsample = self.downsample(x)
        if self.in_ch == 64 // 4:
            downsample = self.equal(x)
            
        seblock = self.seblock(B_conv3)
        out = seblock + downsample
        out = self.relu(out)
        return out
        
class Resnet(nn.Module):
    def __init__(self, layer = [3, 4, 6, 3], num_classes = 10):
        super(Resnet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self.make_layer(SEBlock, in_ch= 64, out_ch= 64, iter=layer[0])
        self.conv3 = self.make_layer(SEBlock, in_ch=128, out_ch=128, iter=layer[1])
        self.conv4 = self.make_layer(SEBlock, in_ch=256, out_ch=256, iter=layer[2])
        self.conv5 = self.make_layer(SEBlock, in_ch=512, out_ch=512, iter=layer[3])
        self.averagepool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(2048, num_classes)
    
    def make_layer(self, in_ch, out_ch, iter):
        layer = []
        if in_ch == 64:
            layer.append(SEBlock(in_ch=in_ch // 4, out_ch=out_ch, down=False))
        else:
            layer.append(SEBlock(in_ch=in_ch * 2, out_ch=out_ch, down=True))
        for _ in range(1, iter):
            layer.append(SEBlock(in_ch=in_ch, out_ch=out_ch, down=False))
            
        return nn.Sequential(*layer)
    
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        
        avgpool = self.averagepool(conv5)
        avgpool = avgpool.reshape(avgpool.size(0), -1)
        linear = self.linear(avgpool)
        
        return linear