import torch
import torch.nn.functional as F
from models.darcy import cc1
from models.darcy import cc2
from models.darcy import bc
import numpy as np
from utils.image_gradient import SobelFilter
from FEA_simp import ComputeTarget

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.loadtxt("data/rho20x40_SIMP_Edge.txt", dtype=np.float32)
# [bs, 1, 20, 40]
data = data.reshape(-1,1,40,20).transpose([0,1,3,2])

data_u = np.loadtxt("data/dis20x40_SIMP_Edge.txt", dtype=np.float32)
data_s = np.loadtxt("data/stress20x40_SIMP_Edge.txt", dtype=np.float32)

ref_u0 = torch.from_numpy(data_u).unsqueeze(1).to(device)
ref_uy = ref_u0[:,:,range(1,1722,2)]
ref_ux = ref_u0[:,:,range(0,1722,2)]
ref_s0 = torch.from_numpy(data_s).unsqueeze(1).to(device)
ref_sx = ref_s0[:,:,range(0,2583,3)]
ref_sy = ref_s0[:,:,range(1,2583,3)]
ref_sxy = ref_s0[:,:,range(2,2583,3)]
# [bs, 5, 21, 41]
ref = torch.cat([ref_ux, ref_uy, ref_sx, ref_sy, ref_sxy],1).view(-1,5,41,21).permute(0,1,3,2)


# input = np.array(np.random.random([1,1,20,40]), dtype=np.float32)

# output = torch.from_numpy(np.array(ComputeTarget(input), dtype=np.float32)).to(device)
# input = torch.from_numpy(input).to(device)
input = torch.from_numpy(data[:1024])
output = ref[:1024]

# post output
WEIGHTS_2x2 = torch.FloatTensor( np.ones([1,1,2,2])/4 ).to(device)
o0 = F.conv2d(output[:,[0]], WEIGHTS_2x2, stride=1, padding=0, bias=None)
o1 = F.conv2d(output[:,[1]], WEIGHTS_2x2, stride=1, padding=0, bias=None) 
o2 = F.conv2d(output[:,[2]], WEIGHTS_2x2, stride=1, padding=0, bias=None) 
o3 = F.conv2d(output[:,[3]], WEIGHTS_2x2, stride=1, padding=0, bias=None) 
o4 = F.conv2d(output[:,[4]], WEIGHTS_2x2, stride=1, padding=0, bias=None) 
output_post = torch.cat([o0,o1,o2,o3,o4],1)

sobel_filter = SobelFilter(64, correct=False, device=device)

print (cc1(input, output, output_post, sobel_filter, device))
print(cc2(output_post, sobel_filter))
loss_boundary = bc(output, output_post)
print(loss_boundary)