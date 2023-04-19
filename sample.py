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

device = torch.device('cuda')

diffuser_opts = {
    'normalization_groups': 32,
    'in_channels': 1,
    'out_channels': 1,
    'channels': 32,
    'num_head_channels': 8,
    'num_residuals': 6,
    'channel_multiple_schedule': [1, 2],
    'interior_attention': 1,
}

model = Diffuser(**diffuser_opts).to(device)

model.load_state_dict(torch.load('model_2.pth'))
model.eval()
# %%
def langevin(m, x0, dt, timesteps, steps=100):
    dist = torch.distributions.MultivariateNormal(torch.zeros(28*28), torch.eye(28*28))

    x = x0
    for i in tqdm.tqdm(range(steps)):
        mx = m(x, timesteps)
        
        bt = dist.sample().reshape(1, 1, 28, 28).to(device)
        dx = 0.5*mx*dt + np.sqrt(dt)*bt

        #import ipdb; ipdb.set_trace()

        x = x + dx

        del bt
        del dx

        if i % 100 == 99:
            plt.show(plt.imshow(x[0, 0].cpu().numpy()))
    return x
# %%
m = torch.distributions.MultivariateNormal(torch.zeros(28*28), torch.eye(28*28))
# %%
start = dataset.test_dataset[2]['pixels'].to(device).unsqueeze(0)
#start = m.sample().reshape(1, 1, 28, 28).to(device)
fig = plt.imshow(start.cpu().numpy()[0][0])
plt.show(fig)

timesteps = torch.tensor([1]).to(device)

with torch.no_grad():
    end = langevin(model, start, 5e-5, timesteps, steps=1000)
# %%

import ipdb; ipdb.set_trace()