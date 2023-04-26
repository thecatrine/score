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
import mnist_dataset as mnist_dataset

import einops
import train_utils

device = torch.device('cuda')

CHANNELS = 3
MODEL_FILE = 'model_latest_epoch.pth'
#MODEL_FILE = 'model_latest_mnist.pth'

MAX_SIGMA = 50.0
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


def continuous(m, samples, steps):
    dt = -1 / steps

    x = samples.clone()

    for t in tqdm.tqdm(np.linspace(1, 0, steps)):
        
        # DW is normal centered at 0 with std = sqrt(dt)
        dw = ( torch.randn_like(x) * np.sqrt(-dt) ).to(device)

        # Rescale score by 1/ sigmas
        score = m(x, t*torch.ones((samples.shape[0],)).to(device)) / sigma(t)


        gt = diffusion(t)
        dx = -1.0*(gt**2)*score*dt + gt*dw

        #import pdb; pdb.set_trace()

        x = x + dx

    return x


def sigma(t):
    return MIN_SIGMA * (MAX_SIGMA / MIN_SIGMA) ** t
    #C = (B-A)*t + A

    #return np.exp(C)

B = np.log(MAX_SIGMA)
A = np.log(MIN_SIGMA)

def dsigmasquared(t):
    return 2*(B-A)*sigma(t)

# Taken from code in paper.
# Had a mismatch between g(t) and g(t)^2 before after rewrite
def diffusion(t):
    s = sigma(t)
    diffusion = s * torch.sqrt(torch.tensor(2 * (B - A),
                                                device=device))
    
    return diffusion


def toimage(foo):
    return Image.fromarray(einops.rearrange(((foo)*256).to(torch.uint8), 'c x y -> x y c').cpu().numpy())

DIM = 5

start = torch.rand((DIM**2, CHANNELS, 32, 32)).to(device)
fig = plt.imshow(start.cpu().numpy()[0][0])
plt.show(fig)
# %%
def sample_from(start):
    with torch.no_grad():
        end = continuous(model, start.to(device), 100)
    return end

end = sample_from(start)
# %%
import loaders.loader_utils as utils

end = end
columns = []
row = []
for im in end:
    row.append(einops.rearrange(im, 'c x y -> x y c'))
    if len(row) == DIM:
        columns.append(torch.cat(row, dim=1))
        #import pdb; pdb.set_trace()
        row = []

all_ims = (torch.cat(columns, dim=0).cpu()*256).to(torch.uint8).numpy()
print(all_ims.shape)
Image.fromarray(all_ims, mode='RGB')


# %%
