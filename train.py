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
#import mnist_dataset
#import cifar_dataset
import datasets

import loaders.datasets as ds
import loaders.loader_utils as utils

import train_utils
import config
import einops
# %%
# Get twich dataset
batch_size = 32
data = ds.NewTwitchDataset(path='loaders/twitch', batch_size=batch_size, shuffle=True, num_workers=8)
loaders = next(data.dataloaders())

# Is this RGBA?
train_dataloader = loaders['train']
test_dataloader = loaders['val']
# %%

# used_dataset = cifar_dataset

# train_dataloader = DataLoader(
#     used_dataset.train_dataset,
#     batch_size=batch_size,
#     pin_memory=False,
#     num_workers=0,
#     drop_last=False,
#     shuffle=False,
#     sampler=None,
# )

# test_dataloader = DataLoader(
#     used_dataset.test_dataset,
#     batch_size=batch_size,
#     pin_memory=False,
#     num_workers=0,
#     drop_last=False,
#     shuffle=False,
#     sampler=None,
# )

writer = SummaryWriter()

# %%
MAX_SIGMA = 50
MIN_SIGMA = 0.01
def sigma(t):
    B = np.log(MAX_SIGMA)
    A = np.log(MIN_SIGMA)

    C = (B-A)*t + A

    return torch.exp(C)

# %%
def denoising_score_estimation(score_net, samples, timesteps):
    sigmas = sigma(timesteps)
    #import ipdb; ipdb.set_trace()

    reshaped_sigmas = sigmas.reshape(samples.shape[0], 1, 1, 1)

    z = torch.randn_like(samples)
    noise = z*reshaped_sigmas
    
    # Rescale output of score net by 1/sigma
    scores = score_net(samples + noise, timesteps) / reshaped_sigmas

    loss = 0.5 * torch.square(scores*reshaped_sigmas + z)

    return loss.mean()


EPOCHS = 3

model, optimizer = config.model_optimizer()
device = config.device

def load_model():
    global model, optimizer
    loaded = torch.load('model_latest_loss.pth')
    #import ipdb; ipdb.set_trace()

    model.load_state_dict(loaded['model'])
    optimizer.load_state_dict(loaded['optimizer'])
load_model()

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, 
    lambda e: train_utils.scheduler_function(EPOCHS*len(train_dataloader), 0, e),
)

scaler = torch.cuda.amp.GradScaler()
lowest_loss = 10e10

fixed_im = None

for epoch in range(EPOCHS):
    loader = iter(train_dataloader)
    for i, batch in enumerate(tqdm.tqdm(loader)):
        #if fixed_im is None:
        #    fixed_im = batch[0]
        #pixels = batch['pixels'].repeat(1, 3, 1, 1)
        #import ipdb; ipdb.set_trace()
#        pixels = batch['pixels']
        pixels = batch[0]
        
        #import ipdb; ipdb.set_trace()
        #pixels = fixed_im.repeat(batch_size, 1, 1, 1)
        #if i == 0:
            #fig = plt.imshow(utils.tensor_to_image(pixels[0]))
            #fig = plt.imshow(rescale(pixels[0]))
            #plt.show(fig)

        # I hope it works to just bias the random rather than doing more math

        #import ipdb; ipdb.set_trace()

        timesteps = torch.rand((pixels.shape[0],))

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # loss, _, _ = sliced_score_estimation_vr(
            #     model, 
            #     pixels.to(device), 
            #     timesteps.to(device),
            # )
            loss = denoising_score_estimation(model, pixels.to(device), timesteps.to(device))
            #loss = alternate_estimation(model, pixels.to(device), timesteps.to(device))
            #loss = anneal_dsm_score_estimation(model, pixels.to(device), ints, levels.to(device))
        
        #loss = 0.5*loss * timesteps.to(device)**2

        #loss = loss.mean()

        writer.add_scalar("loss/train", loss.mean().item(), i)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        writer.add_scalar('lr', scheduler.get_last_lr()[0], i)

        if i % 100 == 99:
            batch = next(iter(test_dataloader))
            with torch.no_grad():
                test_loss = denoising_score_estimation(model, batch[0].to(device), timesteps.to(device)).mean().item()

            writer.add_scalar('loss/test', test_loss, i)
            
            if i % 1000 == 999:
                if test_loss < lowest_loss:
                    print("Saving with test loss", test_loss)
                    train_utils.save_state(
                        epoch, test_loss, model, optimizer, scheduler, f"model_latest_loss.pth"
                    )

                    lowest_loss = test_loss
    
    #train_utils.save_state(
    #    epoch, test_loss, model, optimizer, scheduler, f"model_latest_epoch.pth"
    #)



# %%

# %%
