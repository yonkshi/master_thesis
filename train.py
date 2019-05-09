from datetime import datetime

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

n_iter = 0
dt = datetime.today().strftime('%b-%d') + '-' + generate_slug(2)
writer = SummaryWriter('logs_v2/%s' % dt)
def train():
    global n_iter
    torch.manual_seed(1337)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init the networks
    maneuNet = ManeuvorNetwork(action_space= 5, maneuveur_capacity=15).to(device)
    symEncoder = SymbolEncoder().to(device)
    symDecoder = SymbolDecoder().to(device)
    simulator = SimulatorNet(maneuveur_capacity= 15, symbol_space=19, symbol_capacity= 128, ).to(device)

    params = list(maneuNet.parameters()) + list(symEncoder.parameters()) + list(symDecoder.parameters()) + list(simulator.parameters())
    optimizer = optim.Adam(params)

    n_iter = 0
    for ep in range(1000):
        ds = HDF5SimpleDataset('data/SymWorld_basic.hdf5', ep)
        data_loader = torch.utils.data.DataLoader(ds,
                                    batch_size=32)
        # state var used for maneuver LSTM
        print(ds.n_images,'steps in this episode', ds.n_images/16, 'iterations')
        # random sample a step to collect tensorboard data
        print_step = np.random.randint(len(data_loader))

        maneuNet.reset_state()
        for iteration, batch in enumerate(data_loader):
            n_iter += 1
            # Building the model
            action, s_t0, s_t1, reward  = batch['action_set'].to(device), \
                                              batch['obs_set'].to(device), \
                                              batch['obs_set_t1'].to(device), \
                                              batch['reward_set'].to(device)

            optimizer.zero_grad()

            m = maneuNet(action)
            o_t0 = symEncoder(s_t0)
            o_t1_predicted, reward_out = simulator(o_t0, m)

            s_decoded = symDecoder(o_t0)

            # Observation at T1
            o_t1 = symEncoder(s_t1).detach()


            # Building the loss
            L_symbol = F.binary_cross_entropy_with_logits(o_t1_predicted, o_t1) #
            L_reward = F.mse_loss(reward_out, reward)
            L_ae = F.mse_loss(s_decoded, s_t0)


            loss = L_ae + L_symbol + L_reward

            writer.add_scalar('loss_detail/o_map', L_symbol, n_iter)
            writer.add_scalar('loss_detail/reward', L_reward, n_iter)
            writer.add_scalar('loss_detail/autoencoder', L_ae, n_iter)
            writer.add_scalar('loss/compound', loss, n_iter)

            # only run heavy duty stuff once per episode
            if iteration == print_step:
                visualize_omap(s_t0, o_t0)
                visualize_ae(s_t0, s_decoded)
                visualize_weights(maneuNet=maneuNet, symEncoder=symEncoder, symDecoder=symDecoder, simulator=simulator)

            print('iter', n_iter, 'batch', iteration, 'loss: {:10.4f},'.format(float(loss)),
                  'o_map_loss: {:.4f},'.format(float(L_symbol)),
                  'reward_loss: {:.4f},'.format(float(L_reward)),
                  'ae_loss: {:.4f},'.format(float(L_ae)),
                  )
            print()
            loss.backward(retain_graph=True)
            optimizer.step()



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

def visualize_ae(obs_t0, decoded):

    fig = plt.figure(figsize =(13,7))
    fig.tight_layout()
    plt.tight_layout()

    # plot raw observation
    ax = fig.add_subplot(1,2,1)
    ax.set_title('Raw Input')
    img = obs_t0[0, ...].detach().permute(1, 2, 0).cpu().numpy()  # (X, Y, C)
    ax.imshow(img)
    # Disable axis
    ax.set_xticks([])
    ax.set_yticks([])

    # plot symbolic space
    ax = fig.add_subplot(1,2,2)
    img2 = decoded[0, ...].detach().permute(1, 2, 0).cpu().numpy()  # (X, Y, C)
    ax.imshow(img2)
    # Disable axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Autoencoder output')

    writer.add_figure('Reconstruction', fig, n_iter)

def visualize_weights(**kwargs):
    #extracting params
    for name, network in kwargs.items():
        for param_name, param in network.named_parameters():
            writer.add_histogram('W_{}/{}'.format(name, param_name), param, n_iter)




if __name__ == '__main__':
    train()