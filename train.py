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
# %%

writer = SummaryWriter()

def sliced_score_estimation_vr(score_net, samples, timesteps, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)

    grad1 = score_net(dup_samples, timesteps)
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    grad2 = torch.autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)


    loss = loss1 + loss2
    return loss, loss1, loss2

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

def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, anneal_power=2.):
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)

batch_size = 32
train_dataloader = DataLoader(
    dataset.train_dataset,
    batch_size=batch_size,
    pin_memory=False,
    num_workers=0,
    drop_last=False,
    shuffle=False,
    sampler=None,
)

device = torch.device("cuda")

diffuser_opts = {
        'normalization_groups': 32,
        'in_channels': 1,
        'out_channels': 1,
        'channels': 256,
        'num_head_channels': 64,
        'num_residuals': 6,
        'channel_multiple_schedule': [1, 2, 3],
        'interior_attention': 1,
    }

model = Diffuser(**diffuser_opts).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

scaler = torch.cuda.amp.GradScaler()

for epoch in range(100):
    loader = iter(train_dataloader)
    for i, batch in enumerate(tqdm.tqdm(loader)):
        pixels = batch['pixels']
        if i == 0:
            fig = plt.imshow(pixels[0].squeeze().numpy())
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

        writer.add_scalar("loss", loss.mean().item(), i)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.save(model.state_dict(), f"model_latest.pth")



# %%

# %%
