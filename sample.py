# %% 
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

#import plotly.express as px
#import plotly.figure_factory as ff

import numpy as np
import matplotlib.pyplot as plt

import tqdm

from PIL import Image

from score_model import Diffuser
import dataset

import einops
import train_utils

device = torch.device('cuda')

CHANNELS = 3
MODEL_FILE = 'model_latest.pth'
#MODEL_FILE = 'model_latest_mnist.pth'

MAX_SIGMA = 1.0
MIN_SIGMA = 0.01

diffuser_opts = {
    'normalization_groups': 32,
    'channels': 256,
    'in_channels': CHANNELS,
    'out_channels': CHANNELS,
    'num_head_channels': 64,
    'num_residuals': 6,
    'channel_multiple_schedule': [1, 2, 3],
    'interior_attention': 1,
}

model = Diffuser(**diffuser_opts).to(device)

saved = torch.load(MODEL_FILE)

model.load_state_dict(saved['model'])
model.eval()

# %%

def continuous(m, samples, t_max, t_min, steps):
    dt = (t_min - t_max) / steps

    x = samples.clone()

    for t in tqdm.tqdm(np.linspace(t_max, t_min, steps)):
        dw = (torch.randn_like(x)*dt).to(device)
        score = m(x, t*torch.ones((samples.shape[0],)).to(device))

        gt2 = dsigmasquared(t)
        dx = -1.0*gt2*score*dt + np.sqrt(gt2)*dw

        #import pdb; pdb.set_trace()

        x = x + dx

    return x


def dsigmasquared(t):
    B = MAX_SIGMA
    A = MIN_SIGMA
    return np.exp(2*(np.log(B/A)*t + np.log(A)))*2*np.log(B/A)


start = torch.rand((5, CHANNELS, 32, 32)).to(device)#*MAX_SIGMA
fig = plt.imshow(start.cpu().numpy()[0][0])
plt.show(fig)
# %%
with torch.no_grad():
    end = continuous(model, start.to(device), 1.0, 0.01, 100)
# %%
import loaders.loader_utils as utils

end = end
columns = []
row = []
for im in end:
    #if CHANNELS == 1:
    plt.show(plt.imshow(train_utils.mnist_rescale(im)))
    #else:
    #    plt.show(plt.imshow(utils.tensor_to_image(im.cpu())))
#     row.append(einops.rearrange(im, 'c x y -> x y c'))
#     if len(row) == 7:
#         columns.append(torch.cat(row, dim=1))
#         row = []

# all_ims = torch.cat(columns, dim=0).cpu().numpy()*255
# Image.fromarray(all_ims, mode='RGB')


# %%
