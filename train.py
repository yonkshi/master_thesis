from datetime import datetime
import time
from collections import OrderedDict

import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from coolname import generate_slug
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models import ManeuvorNetwork, SymbolEncoder, SimulatorNet, SymbolDecoder
from data_manager import HDF5SimpleDataset

BATCH_SIZE = 32
SYMBOL_CAPACITY = 64
SYMBOL_X_Y = 20
n_iter = 0
dt = datetime.today().strftime('%b-%d') + '-' + generate_slug(2)
writer = SummaryWriter('logs_v2/%s' % dt)
def train():
    global n_iter
    torch.manual_seed(1337)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init the networks
    maneuNet = ManeuvorNetwork(action_space= 5, maneuveur_capacity=15).to(device)
    symEncoder = SymbolEncoder(symbol_capacity = SYMBOL_CAPACITY).to(device)
    symDecoder = SymbolDecoder(symbol_capacity = SYMBOL_CAPACITY).to(device)
    simulator = SimulatorNet(maneuveur_capacity= 15, symbol_space=SYMBOL_X_Y, symbol_capacity= SYMBOL_CAPACITY, ).to(device)

    params = list(maneuNet.parameters()) + list(symEncoder.parameters()) + list(symDecoder.parameters()) + list(simulator.parameters())
    sim_optim = optim.Adam(params)

    params = list(symEncoder.parameters()) + list(symDecoder.parameters())
    ae_optim = optim.Adam(params, lr=1e-5, weight_decay=1e-7)

    n_iter = 0
    data = HDF5SimpleDataset('data/SymWorld_basic.hdf5')
    for ep in range(1000):
        data.set_episode(ep)
        data_loader = torch.utils.data.DataLoader(data,
                                    batch_size=BATCH_SIZE)
        # state var used for maneuver LSTM
        print(data.ep_length,'steps in this episode', np.ceil(data.ep_length/BATCH_SIZE), 'iterations')
        maneuNet.reset_state()
        for iteration, batch in enumerate(data_loader):
            print('1')
            n_iter += 1
            # Building the model
            action, s_t0, s_t1, reward  = batch['action_set'].to(device), \
                                              batch['obs_set'].to(device), \
                                              batch['obs_set_t1'].to(device), \
                                              batch['reward_set'].to(device)

            sim_optim.zero_grad()
            ae_optim.zero_grad()

            print(2)
            m = maneuNet(action)
            o_t0 = symEncoder(s_t0)
            # o_t1_predicted, reward_out = simulator(o_t0, m)
            # s_decoded = symDecoder(o_t0)
            # ae_loss = F.mse_loss(s_decoded, s_t0)
            visualize_weights(maneuNet=maneuNet, symEncoder=symEncoder, symDecoder=symDecoder, simulator=simulator)
            ae_losses, ae_inputs, ae_outputs = local_vae_loss(symDecoder, o_t0, s_t0, ae_optim)
            ae_loss_mean = torch.mean(ae_losses)

            ae_out_mean = torch.mean(ae_outputs)
            ae_out_var = torch.var(ae_outputs)


            # Observation at T1
            o_t1 = symEncoder(s_t1).detach()
            print(3)

            visualize_ae(ae_inputs, ae_outputs, ae_losses)  # FIXME Temporarily placed here

            # [N,1] -> [N]
            # reward_out = reward_out[:,0]

            # Building the loss
            L_symbol = 0 # F.mse_loss(o_t1_predicted, o_t1) #
            L_reward = 0 # F.mse_loss(reward_out, reward)
            loss = 0 #  L_symbol + L_reward
            print(4)
            # loss.backward(retain_graph=True)
            # sim_optim.step()

            # Losses for each slice of autoencoder
            print(5)

            # for i, ae_loss in enumerate(ae_losses):
            #     ae_optim.zero_grad()
            #     print('loss #', i, 'loss: {:.4f},'.format(float(ae_loss)))
            #     ae_loss.backward(retain_graph=True)
            #     ae_optim.step()
            print(6)


            print(7)
            writer.add_scalar('loss_detail/o_map', L_symbol, n_iter)
            writer.add_scalar('loss_detail/reward', L_reward, n_iter)
            writer.add_scalar('loss_detail/autoencoder', ae_loss_mean, n_iter)
            writer.add_scalar('debug/autoencoder_out_mean', ae_out_mean, n_iter)
            writer.add_scalar('debug/autoencoder_out_var', ae_out_var, n_iter)

            writer.add_scalar('loss/compound', loss, n_iter)

            print('iter', n_iter, 'batch', iteration, 'loss: {:.4f},'.format(float(loss)),
                  'o_map_loss: {:.4f},'.format(float(L_symbol)),
                  'reward_loss: {:.4f},'.format(float(L_reward)),
                  'ae_loss: {:.4f},'.format(float(ae_loss_mean)),
                  )
            print(8)

            if n_iter % 2 == 0:
                visualize_omap(s_t0, o_t0)
                # visualize_ae(ae_inputs, ae_outputs)
                # visualize_weights(maneuNet=maneuNet, symEncoder=symEncoder, symDecoder=symDecoder, simulator=simulator)
                print('>>>>>> 4.5')



        print('episode {} completed'.format(ep))

    writer.close()

def visualize_omap(s_t0, o_t0):

    # Visualize as a matplotlib figure
    idx = np.random.randint(o_t0.size(0))
    sampled_omap = o_t0[idx, ...]
    heatmap = torch.sum(sampled_omap, dim=0).detach().cpu()
    layer_utilization = sampled_omap.sum(1).sum(1).cpu().detach().numpy()

    gs = gridspec.GridSpec(5, 6)

    fig = plt.figure(figsize = (10,7))
    fig.tight_layout()
    plt.tight_layout()

    # plot raw observation
    ax = fig.add_subplot(gs[:4, :3])
    ax.set_title('observation (50x50)')
    img = s_t0[0, ...].detach().permute(1, 2, 0).cpu() # (X, Y, C)
    ax.imshow(img)
    # Disable axis
    ax.set_xticks([])
    ax.set_yticks([])

    # plot symbolic space
    ax = fig.add_subplot(gs[:4, 3:])
    im = ax.imshow(heatmap, cmap='Reds')
    # Disable axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('symbolic space (19x19)')

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    # Plot c
    ax = fig.add_subplot(gs[4, 3:])
    ax.set_xticks([])
    ax.set_xlabel('- 128 Channels ->')
    ax.set_title('Perception Channel Utilization')
    x = np.arange(0, len(layer_utilization)) * 2
    ax.bar(x, layer_utilization)

    writer.add_figure('Learning progress', fig, n_iter)

    # Histogram
    o_t0 =  o_t0.detach().cpu().numpy()

    writer.add_histogram('symbolic_map/overall', o_t0, n_iter)
    for n in range(o_t0.shape[1]):
        writer.add_histogram('symbolic_map_per_layer/{}'.format(n), o_t0[:, n, ...], n_iter)

    for i in range(o_t0.shape[2]):
        for j in range(o_t0.shape[3]):
            writer.add_histogram('symbolic_map_per_cell/{}x{}'.format(i,j), o_t0[..., i, j], n_iter)

def visualize_ae(sliced_inputs, sliced_out, ae_losses):

    idx_i, idx_j = np.random.randint(sliced_inputs.size(0), size=2) # sample a slice
    print('sliced_inputs size',sliced_inputs.size(), 'sliced_out size', sliced_out.size())
    obs_t0 = sliced_inputs[idx_i, idx_j, ...] # randomly sample a region to plot
    decoded = sliced_out[idx_i, idx_j, ...]

    fig = plt.figure(figsize =(13,7))
    fig.tight_layout()
    plt.tight_layout()

    # plot raw observation
    ax = fig.add_subplot(1,3,1)
    ax.set_title('Raw Input')
    img = obs_t0[0, ...].detach().permute(1, 2, 0).cpu().numpy()  # (X, Y, C)
    ax.imshow(img)
    # Disable axis
    ax.set_xticks([])
    ax.set_yticks([])

    # plot symbolic space
    ax = fig.add_subplot(1,3,2)
    img2 = decoded[0, ...].detach().permute(1, 2, 0).cpu().numpy()  # (X, Y, C)
    ax.imshow(img2)
    # Disable axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Autoencoder output')

    # plot symbolic space
    ax = fig.add_subplot(1,3,3)
    img3 = ae_losses.detach().cpu().numpy()  # (X, Y, C)
    im = ax.imshow(img3)
    # Disable axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Autoencoder losses per grid')
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    writer.add_figure('Reconstruction', fig, n_iter)

def visualize_weights(**kwargs):
    #extracting params
    for name, network in kwargs.items():
        for param_name, param in network.named_parameters():
            writer.add_histogram('W_{}/{}'.format(name, param_name), param, n_iter)

def local_vae_loss(decoder:SymbolDecoder, o_t0:torch.Tensor, s_t0:torch.Tensor, ae_optim):

    # See my notebook, this is to evenly slice inputs
    B, C, H, W = o_t0.size()
    stride = np.int(np.floor(s_t0.size(-1) / H))
    ker = np.int(np.mod(s_t0.size(-1),  H) + stride)

    # Merge all x and y into batch
    obj = o_t0.permute(2, 3, 0, 1) # [H, W, B, C]

    # obj = obj.contiguous().view(B * H * W, C) # [ H * W * B, C]
    obj = obj[...,None, None]
    # decoded_batch =  decoder(obj) # [B * W * H, C, 1, 1]
    # decoded = decoded_batch.contiguous().view(H, W, B, 3, ker, ker) # [H, W, B, 3, ker, ker]
    input_sliced = s_t0.unfold(2, ker, stride).unfold(3, ker, stride).permute(2,3,0,1,4,5) # [H, W, B, 3, ker, ker]
    # input_batch = input_sliced.contiguous().view(B*H*W,3, ker, ker )
    # loss = F.mse_loss(decoded_batch, input_batch)
    # loss.backward(retain_graph=True)
    # ae_optim.step()

    decoded = torch.ones((H,W, B, 3, ker, ker))
    losses = torch.zeros((H,W))
    h_arange = torch.randperm(H)
    w_range = torch.randperm(W)

    prev_max = 0
    print('computing autoencoder loss ')
    for _i, i in enumerate(h_arange):
        for _j, j in enumerate(w_range):
            ae_optim.zero_grad()

            decoded_batch = decoder(obj[i,j,...])
            decoded[i,j] = decoded_batch
            loss = F.mse_loss(decoded_batch, input_sliced[i,j])
            loss.backward(retain_graph=True)
            ae_optim.step()
            losses[i,j] = float(loss)

            decoded_detach = decoded_batch.detach().cpu().numpy()

            pc = (_i * H + _j) / (H * W) * 100 # percentage completed
            if pc >= prev_max:
                print('\t\t\t{:.0f}%'.format(pc), end="\r")
                prev_max = np.ceil(pc)
            # print('mean',np.mean(decoded_detach), 'max',np.max(decoded_detach), 'min',np.min(decoded_detach))
    print('')



    return losses,  input_sliced, decoded,







if __name__ == '__main__':
    train()