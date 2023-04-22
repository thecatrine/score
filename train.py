# %% 
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

#import plotly.express as px
#import plotly.figure_factory as ff

import numpy as np
import matplotlib.pyplot as plt

import tqdm

from PIL import Image

from score_model import Diffuser

# %%
import dataset

import loaders.datasets as ds
import loaders.loader_utils as utils

import train_utils
import config
# %%
# Get twich dataset
batch_size = 32
data = ds.NewTwitchDataset(path='loaders/small32', batch_size=batch_size, shuffle=True, num_workers=8)
loaders = next(data.dataloaders())

train_dataloader = loaders['train']
test_dataloader = loaders['val']
# %%

writer = SummaryWriter()

# %%
def sigma(t):
    B = np.log(1)
    A = np.log(0.01)

    C = (B-A)*t + A

    return torch.exp(C)

# %%

def denoising_score_estimation(score_net, samples, timesteps):
    sigmas = sigma(timesteps)
    #import ipdb; ipdb.set_trace()

    reshaped_sigmas = sigmas.reshape(samples.shape[0], 1, 1, 1)

    noise = torch.randn_like(samples)*reshaped_sigmas
    target = (-1 * noise) / (reshaped_sigmas ** 2)
    
    scores = score_net(samples + noise, timesteps)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)

    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * sigmas.squeeze() ** 2

    return loss.mean(dim=0)

model, optimizer = config.model_optimizer()
device = config.device

scaler = torch.cuda.amp.GradScaler()
lowest_loss = 10e10

for epoch in range(100):
    loader = iter(train_dataloader)
    for i, batch in enumerate(tqdm.tqdm(loader)):
        #pixels = batch['pixels']
        #import ipdb; ipdb.set_trace()
        pixels = batch[0]
        if i == 0:
            fig = plt.imshow(utils.tensor_to_image(pixels[0]))
            #plt.show(fig)

        # I hope it works to just bias the random rather than doing more math


        timesteps = torch.rand((pixels.shape[0],))

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # loss, _, _ = sliced_score_estimation_vr(
            #     model, 
            #     pixels.to(device), 
            #     timesteps.to(device),
            # )
            loss = denoising_score_estimation(model, pixels.to(device), timesteps.to(device))
            #loss = anneal_dsm_score_estimation(model, pixels.to(device), ints, levels.to(device))
        
        #loss = 0.5*loss * timesteps.to(device)**2

        #loss = loss.mean()

        writer.add_scalar("loss/train", loss.mean().item(), i)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if i % 100 == 99:
            batch = next(iter(test_dataloader))
            with torch.no_grad():
                test_loss = denoising_score_estimation(model, batch[0].to(device), timesteps.to(device)).mean().item()

            writer.add_scalar('loss/test', test_loss, i)
            
            if i % 1000 == 999:
                if test_loss < lowest_loss:
                    train_utils.save_state(model, optimizer, f"model_latest.pth")

                    lowest_loss = test_loss



# %%

# %%
