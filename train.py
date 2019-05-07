from datetime import datetime

from torch import nn, optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from coolname import generate_slug
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import *
from data_manager import HDF5SimpleDataset



datetime.today().strftime('%Y-%m-%d')
EPOCHS = 100


def train():
    torch.manual_seed(1337)

    dt = datetime.today().strftime('%b-%d') + '-' + generate_slug(2)
    writer = SummaryWriter(f'log/{dt}')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO google sync


    maneuNet = ManeuvorNetwork(action_space= 5, maneuveur_capacity=15).to(device)
    symbolNet = SymbolNet().to(device)
    simNet = SimulatorNet(maneuveur_capacity= 15, symbol_space=19, symbol_capacity= 128, ).to(device)

    tb_gen_graph(writer, maneuNet, symbolNet, simNet)

    params = list(maneuNet.parameters()) + list(symbolNet.parameters()) + list(simNet.parameters())
    optimizer = optim.Adam(params)

    n_iter = 0
    for ep in range(1000):
        ds = HDF5SimpleDataset('data/SymWorld_basic.hdf5', ep)
        data_loader = torch.utils.data.DataLoader(ds,
                                    batch_size=16)
        # state var used for maneuver LSTM
        print(ds.n_images,'steps in this episode', ds.n_images/16, 'iterations')

        maneuNet.reset_state()
        for iteration, batch in enumerate(data_loader):
            n_iter += 1
            # Building the model
            action, obs_t0, obs_t1, reward  = batch['action_set'].to(device), \
                                              batch['obs_set'].to(device), \
                                              batch['obs_set_t1'].to(device), \
                                              batch['reward_set'].to(device)

            optimizer.zero_grad()

            m = maneuNet(action)
            o_t0 = symbolNet(obs_t0)
            o_t1_predicted, reward_out = simNet(o_t0, m)

            # Observation at T1
            o_t1 = symbolNet(obs_t1).detach()


            # Building the loss
            L_symbol = F.binary_cross_entropy(o_t1_predicted, o_t1) #
            L_reward = F.mse_loss(reward_out, reward)


            loss = L_symbol + L_reward

            writer.add_scalar('loss/o_map', L_symbol, n_iter)
            writer.add_scalar('loss/reward', L_reward, n_iter)
            writer.add_scalar('loss/compound', loss, n_iter)

            # TODO Print action/maneuver
            # TODO Print progression of omap (both t0 and t1)
            # TODO Show omap as a histogram?

            visualize_omap(writer, obs_t0, o_t0, n_iter)

            print('iter', n_iter, 'batch', iteration, 'loss', "{:10.4f}".format(float(loss)) )
            loss.backward(retain_graph=True)
            optimizer.step()



        print(f'episode {ep} completed')

    writer.close()

def tb_gen_graph(writer, maneuNet, symbolNet, simulator):
    ''' Generates graph for Tensorboard X'''
    dummy_action = torch.rand(1,5)
    dummy_obs = torch.rand(1,3, 50, 50)
    dummy_simulator_m = torch.rand(1, 15)
    dummy_simulator_o = torch.rand(1, 128, 19, 19)

    writer.add_graph(symbolNet, (dummy_obs, ), True)
    # writer.add_graph(simulator, (dummy_simulator_o,dummy_simulator_m), True)

def visualize_omap(writer, obs_t0, omap, n_iter):

    # Visualize as a matplotlib figure
    z = omap.size(0)
    idx = np.random.randint(omap.size(0))
    sampled_omap = omap[idx, ...]
    heatmap = torch.sum(sampled_omap, dim=0).detach()
    layer_utilization = sampled_omap.sum(1).sum(1).detach().numpy()

    gs = gridspec.GridSpec(5, 6)

    fig = plt.figure(figsize = (10,7))
    fig.tight_layout()
    plt.tight_layout()

    # plot raw observation
    ax = fig.add_subplot(gs[:4, :3])
    ax.set_title('observation (50x50)')
    img = obs_t0[0,...].detach().permute(1,2,0) # (X, Y, C)
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





if __name__ == '__main__':
    train()