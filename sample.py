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

device = torch.device('cuda')

diffuser_opts = {
    'normalization_groups': 32,
    'channels': 256,
    'in_channels': 1,
    'out_channels': 1,
    'num_head_channels': 64,
    'num_residuals': 6,
    'channel_multiple_schedule': [1, 2, 3],
    'interior_attention': 1,
}

model = Diffuser(**diffuser_opts).to(device)

model.load_state_dict(torch.load('model_latest.pth'))
model.eval()

levels = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1]
# %%
def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
    images = []

    with torch.no_grad():
        for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * torch.sqrt(step_size * 2)
                grad = scorenet(x_mod, torch.ones(x_mod.shape[0], device=x_mod.device)*sigma)
                x_mod = x_mod + step_size * grad + noise
                # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                #                                                          grad.abs().max()))

        return images


def continuous(m, samples, t_max, t_min, steps):
    t = t_max
    dt = (t_min - t_max) / steps

    x = samples

    for t in tqdm.tqdm(np.linspace(t_max, t_min, steps)):
        dw = (torch.randn_like(x)*dt).to(device)
        score = m(x, t*torch.ones((samples.shape[0],)).to(device))

        gt2 = dsigmasquared(t)
        dx = -1.0*gt2*score*dt + np.sqrt(gt2)*dw

        x = x + dx
        t = t + dt

    return x


def dsigmasquared(t):
    B = 1.0
    A = 0.01
    return np.exp(2*(np.log(B/A)*t + np.log(A)))*2*np.log(B/A)

def langevin(m, x0, dt, steps=100):
    dist = torch.distributions.MultivariateNormal(torch.zeros(28*28), torch.eye(28*28))
    x = x0

    level_0 = levels[0]

    for sigma in reversed(levels):
        frac = (sigma / level_0 ) ** 2
        for i in tqdm.tqdm(range(steps)):
            timesteps = sigma * torch.ones(x.shape[0]).to(device)
            mx = m(x, timesteps)
            
            eps = dt * frac

            bt = dist.sample(torch.Size([x.shape[0]])).reshape(x.shape).to(device)
            dx = 0.5*mx*eps + np.sqrt(eps)*bt

            #import ipdb; ipdb.set_trace()

            x = x + dx

            del bt
            del dx
        
        plt.show(plt.imshow(x[0][0].cpu().numpy()))
    return x
# %%

start = torch.rand((25, 1, 28, 28)).to(device)
fig = plt.imshow(start.cpu().numpy()[0][0])
plt.show(fig)

with torch.no_grad():
    end = continuous(model, start.to(device), 1.0, 0.01, 100)
# %%

end = end
columns = []
row = []
for im in end:
    row.append(torch.rearrange(im, 'cxy->xyc'))
    if len(row) == 5:
        columns.append(torch.cat(row, dim=1))
        row = []

Image.fromarray(torch.cat(columns, dim=0).cpu().numpy()*255).convert("RGB")

# %%
with torch.no_grad():
    end = langevin(model, start, 2e-5, steps=100)

for im in end:
    plt.show(plt.imshow(im[0].cpu().numpy()))
    plt.show(plt.imshow(Image.fromarray(im[0].cpu().numpy()*255)))
# %%

# %%
samples = torch.randn(5, 1, 28, 28).to(device)
sigmas = torch.Tensor(levels).to(device)

images = anneal_Langevin_dynamics(samples, model, sigmas, 100, 0.00002)
# %%
for im in images[-1]:
    plt.show(plt.imshow(im[0].cpu().numpy()))
    plt.show(plt.imshow(Image.fromarray(im[0].cpu().numpy()*255)))
# %%
