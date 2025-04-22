import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_params(shape,init_func):
    return nn.Parameter(init_func(torch.empty(shape)))

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride):
        super(BasicBlock, self).__init__()

        self.is_unfold = False
        self.stride = stride
        self.in_planes=in_planes
        self.planes=planes

        self.dw_weight1    = generate_params((in_planes,1,3,3),nn.init.normal_)
        self.conv1_weight1 = generate_params((planes, in_planes, 1, 1),nn.init.dirac_)

        self.dw_weight2    = generate_params((planes,1,3,3),nn.init.normal_)
        self.conv1_weight2 = generate_params((planes, planes, 1, 1),nn.init.dirac_)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
           self.shortcut = nn.Sequential(
                           nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                           nn.BatchNorm2d(planes)
                            )

    def _reg_loss(self):
        if self.stride==1:
           W = ((self.dw_weight1*self.conv1_weight1)**2).sum() + ((self.dw_weight2*self.conv1_weight2)**2).sum()
        else:
           W = ((self.conv1.weight)**2).sum() + ((self.dw_weight2*self.conv1_weight2)**2).sum()
        return W
    
    def unfold(self):
        self.is_unfold = True

    def forward(self, x):
        
        if self.is_unfold:
           out = F.conv2d(x,self.dw_weight1,stride=self.stride,padding=1,bias=None,groups=self.in_planes) 
           out = F.conv2d(out,self.conv1_weight1,stride=1,padding=0,bias=None) 
        else:
           out = F.conv2d(x,(self.dw_weight1*(self.conv1_weight1.transpose(1,0))).transpose(1,0),stride=self.stride,padding=1,bias=None) 
        out = self.bn1(out)
        out = F.relu(out,True)

        if self.is_unfold:
           out = F.conv2d(out,self.dw_weight2,stride=1,padding=1,bias=None,groups=self.planes) 
           out = F.conv2d(out,self.conv1_weight2,stride=1,padding=0,bias=None)
        else:
           out = F.conv2d(out,(self.dw_weight2*(self.conv1_weight2.transpose(1,0))).transpose(1,0),stride=1,padding=1,bias=None) 
        out = self.bn2(out)
        out += self.shortcut(x)
        return out
    
 
class FoldNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width_scaler=1., expansion=1):
        super(FoldNet, self).__init__()
        

        self.in_planes = int(16*width_scaler)

        self.conv1 = nn.Sequential(nn.Conv2d(3,self.in_planes, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_planes),
                                   nn.ReLU(True))
        
        self.layer1 = self._make_layer(block, int(16*width_scaler), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(32*width_scaler), num_blocks[1], stride=2)
        self.layer3= self._make_layer(block, int(64*width_scaler), num_blocks[2], stride=2)

        self.linear = nn.Linear(int(64*width_scaler), num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes 

        return nn.Sequential(*layers)

    
    def unfold_net(self):
        for sub_module in self.modules():
            if hasattr(sub_module, "unfold"):
                sub_module.unfold()

    def forward(self, x):
        
        out = self.conv1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def foldnet20(num_classes,expansion,width_scaler=1):
    return FoldNet(BasicBlock, [3, 3, 3],num_classes,expansion=expansion,width_scaler=width_scaler)


def foldnet56(num_classes,expansion,width_scaler=1):
    return FoldNet(BasicBlock, [9, 9, 9],num_classes,expansion=expansion,width_scaler=width_scaler)

if __name__ == '__main__':
    net = foldnet56(10,10,10)
    net.eval()
    print(net)
    x = torch.randn(1,3,224,224)
    y1 = net(x)
    net.unfold_net()
    y2 = net(x)
    error = (y1-y2).abs().sum()
    print(error)
