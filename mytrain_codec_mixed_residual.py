"""Physics-constraint surrogates.
Convolutional Encoder-decoder networks for surrogate modeling of darcy flow.
Assume the PDEs and boundary conditions are known.
Train the surrogate with mixed residual loss, instead of maximum likelihood.

5 runs per setup
setup:
    - training with different number of input: 
        512, 1024, 2048, 4096, 8192
    with mini-batch size 8, 8, 16, 32, 32, correpondingly.
    - metric: 
        relative L2 error, i.e. NRMSE 
        R^2 score
    - Other default hyperparameters in __init__ of Parser class
"""
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.codec import DenseED
from models.darcy import conv_constitutive_constraint as constitutive_constraint
from models.darcy import conv_continuity_constraint as continuity_constraint
from models.darcy import conv_boundary_condition as boundary_condition
# my loss
from models.darcy import cc1
from models.darcy import cc2
from models.darcy import cc2new
from models.darcy import bc

from models.darcy import cc1_new, bc_new

from torch.utils.data import DataLoader, TensorDataset

from utils.image_gradient import SobelFilter
from utils.load import load_data
from utils.misc import mkdirs, to_numpy
from utils.plot import plot_prediction_det, save_stats
from utils.practices import OneCycleScheduler, adjust_learning_rate, find_lr
import time
import argparse
import random
from pprint import pprint
import json
import sys
import matplotlib.pyplot as plt
import h5py
from FEA_simp import ComputeTarget
from mytest import testsample, get_testsample

plt.switch_backend('agg')


class Parser(argparse.ArgumentParser):
    def __init__(self): 
        super(Parser, self).__init__(description='Learning surrogate with mixed residual norm loss')
        self.add_argument('--exp-name', type=str, default='codec/mixed_residual', help='experiment name')
        self.add_argument('--exp-dir', type=str, default="./experiments", help='directory to save experiments')      
        # codec
        self.add_argument('--blocks', type=list, default=[8, 12, 8], help='list of number of layers in each dense block')
        self.add_argument('--growth-rate', type=int, default=16, help='number of output feature maps of each conv layer within each dense block')
        self.add_argument('--init-features', type=int, default=48, help='number of initial features after the first conv layer')        
        self.add_argument('--drop-rate', type=float, default=0., help='dropout rate')
        self.add_argument('--upsample', type=str, default='nearest', choices=['nearest', 'bilinear'])
        # data 
        self.add_argument('--data-dir', type=str, default="./datasets", help='directory to dataset')
        self.add_argument('--data', type=str, default='grf_kle512', choices=['grf_kle512', 'channelized'])
        self.add_argument('--ntrain', type=int, default=7003, help="number of training data")
        self.add_argument('--ntest', type=int, default=512, help="number of validation data")
        self.add_argument('--imsize', type=int, default=64)
        # training
        self.add_argument('--run', type=int, default=1, help='run instance')
        self.add_argument('--epochs', type=int, default=50000, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=1e-3, help='learnign rate')
        self.add_argument('--lr-div', type=float, default=2., help='lr div factor to get the initial lr')
        self.add_argument('--lr-pct', type=float, default=0.3, help='percentage to reach the maximun lr, which is args.lr')
        self.add_argument('--weight-decay', type=float, default=0., help="weight decay")
        self.add_argument('--weight-bound', type=float, default=10, help="weight for boundary loss")
        self.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
        self.add_argument('--test-batch-size', type=int, default=64, help='input batch size for testing')
        self.add_argument('--seed', type=int, default=1, help='manual seed used in Tensor')
        self.add_argument('--cuda', type=int, default=0, choices=[0, 1, 2, 3], help='cuda index')
        # logging
        self.add_argument('--debug', action='store_true', default=True, help='debug or verbose')
        self.add_argument('--ckpt-epoch', type=int, default=None, help='which epoch of checkpoints to be loaded')
        self.add_argument('--ckpt-freq', type=int, default=2000, help='how many epochs to wait before saving model')
        self.add_argument('--log-freq', type=int, default=1, help='how many epochs to wait before logging training status')
        self.add_argument('--plot-freq', type=int, default=50, help='how many epochs to wait before plotting test output')
        self.add_argument('--plot-fn', type=str, default='imshow', choices=['contourf', 'imshow'], help='plotting method')

    def parse(self):
        args = self.parse_args()

        hparams = f'{args.data}_ntrain{args.ntrain}_run{args.run}_bs{args.batch_size}_lr{args.lr}_epochs{args.epochs}'
        if args.debug:
            hparams = 'debug/' + hparams
        args.run_dir = args.exp_dir + '/' + args.exp_name + '/' + hparams
        args.ckpt_dir = args.run_dir + '/checkpoints'
        # print(args.run_dir)
        # print(args.ckpt_dir)
        mkdirs(args.run_dir, args.ckpt_dir)

        # assert args.ntrain % args.batch_size == 0 and \
        #     args.ntest % args.test_batch_size == 0

        if args.seed is None:
            args.seed = random.randint(1, 10000)
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        print('Arguments:')
        pprint(vars(args))
        with open(args.run_dir + "/args.txt", 'w') as args_file:
            json.dump(vars(args), args_file, indent=4)

        return args


if __name__ == '__main__':

    args = Parser().parse()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    args.train_dir = args.run_dir + '/training'
    args.pred_dir = args.train_dir + '/predictions'
    mkdirs(args.train_dir, args.pred_dir)

    model = DenseED(in_channels=1, out_channels=2, 
                    imsize=args.imsize,
                    blocks=args.blocks,
                    growth_rate=args.growth_rate, 
                    init_features=args.init_features,
                    drop_rate=args.drop_rate,
                    out_activation=None,
                    upsample=args.upsample).to(device)
    if args.debug:
        # print(model)
        pass
    # if start from ckpt
    if args.ckpt_epoch is not None:
        ckpt_file = args.run_dir + f'/checkpoints/model_epoch{args.ckpt_epoch}.pth'
        model.load_state_dict(torch.load(ckpt_file, map_location='cpu'))
        print(f'Loaded ckpt: {ckpt_file}')
        print(f'Resume training from epoch {args.ckpt_epoch + 1} to {args.epochs}')

    # load data
    if args.data == 'grf_kle512':
        train_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/kle512_lhs10000_train.hdf5'
        test_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/kle512_lhs1000_val.hdf5'
        ntrain_total, ntest_total = 10000, 1000
    elif args.data == 'channelized':
        train_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/channel_ng64_n4096_train.hdf5'
        test_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/channel_ng64_n512_test.hdf5'
        ntrain_total, ntest_total = 4096, 512
    assert args.ntrain <= ntrain_total, f"Only {args.ntrain_total} data "\
        f"available in {args.data} dataset, but needs {args.ntrain} training data."
    assert args.ntest <= ntest_total, f"Only {args.ntest_total} data "\
        f"available in {args.data} dataset, but needs {args.ntest} test data."   
    
    # modif its data
    with h5py.File(train_hdf5_file, 'r') as f:
        # x_data = f['input'][:ndata]
        x_data = f['input'][:args.ntrain,:,:20,:40]
    x_data = (x_data - x_data.min()) /( x_data.max() - x_data.min())
    print("ComputeTarget....")
    # label_data = ComputeTarget(x_data)
    # print("ComputeTarget finish")
    # data_tuple = (torch.FloatTensor(x_data), torch.FloatTensor(label_data).to(device))
    # train_loader = DataLoader(TensorDataset(*data_tuple),
    #     batch_size=args.batch_size, shuffle=True, drop_last=True)

    # train_loader, _ = load_data(train_hdf5_file, args.ntrain, args.batch_size, 
    #     only_input=True, return_stats=False)


    test_loader, test_stats = load_data(test_hdf5_file, args.ntest, 
        args.test_batch_size, only_input=False, return_stats=True)
    y_test_variation = test_stats['y_variation']
    print(f'Test output variation per channel: {y_test_variation}')

    data = np.loadtxt("data/rho20x40_SIMP_Edge.txt", dtype=np.float32)
    # [bs, 1, 20, 40]
    data = data.reshape(-1,1,40,20).transpose([0,1,3,2])

    data_u = np.loadtxt("data/dis20x40_SIMP_Edge.txt", dtype=np.float32)
    data_s = np.loadtxt("data/stress20x40_SIMP_Edge.txt", dtype=np.float32)

    # filter data
    result_label = []
    for ind,i in enumerate(data_u):
        if i.max()>5000 or i.min()<-1000:
            pass
        else:
            result_label.append(ind)
    data = data[result_label]
    data_u = data_u[result_label]
    data_s = data_s[result_label]

    ref_u0 = torch.from_numpy(data_u).unsqueeze(1).to(device)
    ref_uy = ref_u0[:,:,range(1,1722,2)]
    ref_ux = ref_u0[:,:,range(0,1722,2)]
    ref_s0 = torch.from_numpy(data_s).unsqueeze(1).to(device)
    ref_sx = ref_s0[:,:,range(0,2583,3)]
    ref_sy = ref_s0[:,:,range(1,2583,3)]
    ref_sxy = ref_s0[:,:,range(2,2583,3)]
    # [bs, 5, 21, 41]
    ref = torch.cat([ref_ux, ref_uy, ref_sx, ref_sy, ref_sxy],1).view(-1,5,41,21).permute(0,1,3,2)

    # data_tuple = (torch.FloatTensor(data),)
    data_tuple = (torch.FloatTensor(data), ref)
    # torch.FloatTensor(data_dis))
    train_loader = DataLoader(TensorDataset(*data_tuple),
        batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)
    scheduler = OneCycleScheduler(lr_max=args.lr, div_factor=args.lr_div, 
                        pct_start=args.lr_pct)
    sobel_filter = SobelFilter(args.imsize, correct=False, device=device)

    # 1/4 conv to make output (21x41) => (20x40)
    WEIGHTS_2x2 = torch.FloatTensor( np.ones([1,1,2,2])/4 ).to(device)
    # post_y = F.conv2d(y_test, WEIGHTS_2x2, stride=1, padding=0, bias=None)

    n_out_pixels = test_loader.dataset[0][1].numel()
    print(f'Number of out pixels per image: {n_out_pixels}')

    logger = {}
    logger['loss_train'] = []
    logger['loss_pde1'] = []
    logger['loss_pde2'] = []
    logger['loss_b'] = []
    logger['u_l2loss'] = []
    logger['s_l2loss'] = []


    print('Start training...................................................')
    start_epoch = 1 if args.ckpt_epoch is None else args.ckpt_epoch + 1
    tic = time.time()
    # step = 0
    total_steps = args.epochs * len(train_loader)
    print(f'total steps: {total_steps}')
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        # if epoch == 30:
        #     print('begin finding lr')
        #     logs,losses = find_lr(model, train_loader, optimizer, loss_fn, 
        #           args.weight_bound, init_value=1e-8, final_value=10., beta=0.98)
        #     plt.plot(logs[10:-5], losses[10:-5])
        #     plt.savefig(args.train_dir + '/find_lr.png')
        #     sys.exit(0)
        relative_l2 = []

        loss_train, mse = 0., 0.
        for batch_idx, (input, target) in enumerate(train_loader, start=1):
            input = input.to(device)
            model.zero_grad()
            output = model(input)
            # post output
            # o0 = F.conv2d(output[:,[0]], WEIGHTS_2x2, stride=1, padding=0, bias=None)
            # o1 = F.conv2d(output[:,[1]], WEIGHTS_2x2, stride=1, padding=0, bias=None) 
            # o2 = F.conv2d(output[:,[2]], WEIGHTS_2x2, stride=1, padding=0, bias=None) 
            # o3 = F.conv2d(output[:,[3]], WEIGHTS_2x2, stride=1, padding=0, bias=None) 
            # o4 = F.conv2d(output[:,[4]], WEIGHTS_2x2, stride=1, padding=0, bias=None) 
            # output_post = torch.cat([o0,o1,o2,o3,o4],1)

            # new
            target = target[:,:2,:,:]
            loss_pde = cc1_new(input, output, device)
            loss_pde1 = loss_pde
            # loss_boundary = 0
            loss_boundary = bc_new(output)

            # old
            # loss_pde1 = cc1(input, output, output_post, sobel_filter, device)
            # # loss_pde2 = cc2(output_post, sobel_filter)
            # loss_pde2 = cc2new(output, sobel_filter)
            # loss_pde = 2*loss_pde1 + 3*loss_pde2
            # loss_boundary = bc(output, output_post)

            loss = 200*loss_pde + loss_boundary
             # * args.weight_bound
            loss.backward()

            # uapr = output.view(input.shape[0],5,-1)
            # e2 = torch.sum((output[:,:2] - target[:,:2]) ** 2 , [-1, -2])
            # torch.sqrt(e2 / (target[:,:2] ** 2).sum([-1, -2]))
            err2_sum = torch.sum((output - target) ** 2, [-1, -2])
            # relative_l2.append(err2_sum)
            relative_l2.append(torch.sqrt(err2_sum / (target ** 2).sum([-1, -2]) ) )
            # lr scheduling
            step = (epoch - 1) * len(train_loader) + batch_idx
            pct = step / total_steps
            lr = scheduler.step(pct)
            adjust_learning_rate(optimizer, lr)
            optimizer.step()
            loss_train += loss.item()

        loss_train /= batch_idx
        relative_l2 = to_numpy(torch.cat(relative_l2, 0).mean(0))
        relative_u = np.mean(relative_l2[:2])
        relative_s = np.mean(relative_l2[2:])
        print(f'Epoch {epoch}, lr {lr:.6f}')
        # print(f'Epoch {epoch}: training loss: {loss_train:.6f}, pde1: {loss_pde1:.6f}, pde2: {loss_pde2:.6f}, '\
        #     # f'dirichlet {loss_dirichlet:.6f}, nuemann {loss_neumann:.6f}')
        #     f'boundary: {loss_boundary:.6f}, relative-u: {relative_u: .5f}, relative_s: {relative_s: .5f}')
        print(f'Epoch {epoch}: training loss: {loss_train:.6f}, pde1: {loss_pde1:.6f} '\
            f'boundary: {loss_boundary:.6f}, relative-u: {relative_u: .5f}')
        if epoch % args.log_freq == 0:
            logger['loss_train'].append(loss_train)
            logger['loss_pde1'].append(loss_pde1)
            # logger['loss_pde2'].append(loss_pde2)
            logger['loss_b'].append(loss_boundary)
            logger['u_l2loss'].append(relative_u)
            logger['s_l2loss'].append(relative_s)
        if epoch % args.ckpt_freq == 0:
            torch.save(model, args.ckpt_dir + "/model_epoch{}.pth".format(epoch))
            sampledir = args.ckpt_dir + "/model{}".format(epoch)
            mkdirs((sampledir))
            get_testsample(model, sampledir, data, ref)
            # testsample(args.ckpt_dir + "/model_epoch{}.pth".format(epoch), sampledir)
        
        # with torch.no_grad():
        #     test(epoch)

    tic2 = time.time()
    print(f'Finished training {args.epochs} epochs with {args.ntrain} data ' \
        f'using {(tic2 - tic) / 60:.2f} mins')
    metrics = ['loss_train','loss_pde1', 'loss_b', 'u_l2loss', 's_l2loss']
    # metrics = ['loss_train','loss_pde1', 'loss_pde2', 'loss_b', 'u_l2loss', 's_l2loss']
    save_stats(args.train_dir, logger, *metrics)
    args.training_time = tic2 - tic
    args.n_params, args.n_layers = model.model_size
    with open(args.run_dir + "/args.txt", 'w') as args_file:
        json.dump(vars(args), args_file, indent=4)
