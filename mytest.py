import torch
import numpy as np
import skimage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

c1 = np.array([1,1,0])
c2 = np.array([0,0,1])

def plot_input(data, filename):
    skimage.io.imsave(filename, data,check_contrastbool=False)

def plot_img(data, filename):
    data = (data - data.min()) / (data.max() - data.min())
    x = data.shape[0]
    y = data.shape[1]
    img = np.zeros([x,y,3])
    for i in range(x):
        for j in range(y):
            img[i,j] = (1-data[i,j])*c1+data[i,j]*c2
    skimage.io.imsave(filename, img,check_contrastbool=False)


def show_hist(axis):
    print ('real data:')
    print (np.histogram(output[:,axis].contiguous().view(-1).cpu().numpy()) )
    print ('predict data:')
    print (np.histogram(output_bak[:,axis].contiguous().view(-1).cpu().detach().numpy()) )

u_max = np.zeros(5)
u_min = np.zeros(5)
u_min[0] = -400
u_max[0] = 500

u_min[1] = -50
u_max[1] = 50
u_min[2] = -1
u_max[2] = 5
u_min[3] = -0.5
u_max[3] = 1
u_min[4] = -0.5
u_max[4] = 0.5

def plot_img_notnorm(data, filename, axis):
    # data = (data - data.min()) / (data.max() - data.min())
    data = (data - u_min[axis]) / (u_max[axis]- u_min[axis])
    data[data>1] = 1
    data[data<0] = 0
    x = data.shape[0]
    y = data.shape[1]
    img = np.zeros([x,y,3])
    for i in range(x):
        for j in range(y):
            # img[i,j] = (1-data[i,j])*c1+data[i,j]*c2
            img[i,j] = data[i,j]*255
    skimage.io.imsave(filename, img,check_contrastbool=False)


model = torch.load('experiments/codec/mixed_residual/debug/grf_kle512_ntrain4'\
    '096_run1_bs32_lr0.001_epochs500/checkpoints/model_epoch500.pth')

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

input = torch.from_numpy(data).to(device)
output = model(input)

sample_num = 20
num_label = np.arange(0, input.shape[0])
np.random.shuffle(num_label)
sample = num_label[:sample_num]

# input_dir = "input_img"
# for i in range(input.shape[0]):
#     plot_input(1-input[i,0].cpu().detach().numpy(),f'{input_dir}/{i}-input.jpg')
#     plot_img(output[i,0].cpu().detach().numpy(),f'{input_dir}/{i}-ux-pre.jpg')
#     plot_img(output[i,1].cpu().detach().numpy(),f'{input_dir}/{i}-uy-pre.jpg')
#     plot_img(output[i,2].cpu().detach().numpy(),f'{input_dir}/{i}-sx-pre.jpg')
#     plot_img(output[i,3].cpu().detach().numpy(),f'{input_dir}/{i}-sy-pre.jpg')
#     plot_img(output[i,4].cpu().detach().numpy(),f'{input_dir}/{i}-sxy-pre.jpg')
#     plot_img(ref[i,0].cpu().detach().numpy(),f'{input_dir}/{i}-ux-ref.jpg')
#     plot_img(ref[i,1].cpu().detach().numpy(),f'{input_dir}/{i}-uy-ref.jpg')
#     plot_img(ref[i,2].cpu().detach().numpy(),f'{input_dir}/{i}-sx-ref.jpg')
#     plot_img(ref[i,3].cpu().detach().numpy(),f'{input_dir}/{i}-sy-ref.jpg')
#     plot_img(ref[i,4].cpu().detach().numpy(),f'{input_dir}/{i}-sxy-ref.jpg')

dirr = "result_img_new"
for i in sample:
    plot_input(1-input[i,0].cpu().detach().numpy(),f'{dirr}/{i}-input.jpg')
    plot_img_notnorm(output[i,0].cpu().detach().numpy(),f'{dirr}/{i}-ux-pre.jpg',0)
    plot_img_notnorm(output[i,1].cpu().detach().numpy(),f'{dirr}/{i}-uy-pre.jpg',1)
    # plot_img(output[i,1].cpu().detach().numpy(),f'{dirr}/{i}-uy-pre.jpg')
    plot_img_notnorm(output[i,2].cpu().detach().numpy(),f'{dirr}/{i}-sx-pre.jpg',2)
    plot_img_notnorm(output[i,3].cpu().detach().numpy(),f'{dirr}/{i}-sy-pre.jpg',3)
    plot_img_notnorm(output[i,4].cpu().detach().numpy(),f'{dirr}/{i}-sxy-pre.jpg',4)
    plot_img_notnorm(ref[i,0].cpu().detach().numpy(),f'{dirr}/{i}-ux-ref.jpg',0)
    plot_img_notnorm(ref[i,1].cpu().detach().numpy(),f'{dirr}/{i}-uy-ref.jpg',1)
    # plot_img(ref[i,1].cpu().detach().numpy(),f'{dirr}/{i}-uy-ref.jpg')
    plot_img_notnorm(ref[i,2].cpu().detach().numpy(),f'{dirr}/{i}-sx-ref.jpg',2)
    plot_img_notnorm(ref[i,3].cpu().detach().numpy(),f'{dirr}/{i}-sy-ref.jpg',3)
    plot_img_notnorm(ref[i,4].cpu().detach().numpy(),f'{dirr}/{i}-sxy-ref.jpg',4)
    # input[sample]