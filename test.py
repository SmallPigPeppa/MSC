
import torch.nn.functional as F
import torch

a=torch.rand([64,3,224,224])
b=F.interpolate(a,size=(224,224),mode='bilinear')
print(a-b)