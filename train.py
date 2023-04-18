# %% 
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import plotly.express as px
import plotly.figure_factory as ff

import numpy as np
import matplotlib.pyplot as plt

import tqdm

from PIL import Image

from score_model import Diffuser

# %%
from datasets import load_dataset

dataset = load_dataset("mnist")

def map_func(examples):
    #import ipdb; ipdb.set_trace()
    examples['pixels'] = []
    for ex in examples['image']:
        im = np.array(ex)
        tensor = torch.Tensor(im)
        examples['pixels'].append(tensor.unsqueeze(0))
    
    return examples

train_dataset = dataset['train'].map(map_func, batched=True).with_format('torch')
test_dataset = dataset['test'].map(map_func, batched=True).with_format('torch')
# %%

def sliced_score_estimation_vr(score_net, samples, timesteps, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)

    grad1 = score_net(dup_samples, timesteps)
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    grad2 = torch.autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()

batch_size = 32
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    pin_memory=False,
    num_workers=0,
    drop_last=False,
    shuffle=False,
    sampler=None,
)

device = torch.device("mps")

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


for epoch in range(1):
    loader = iter(train_dataloader)
    for batch in tqdm.tqdm(loader):
        timesteps = torch.ones(batch['pixels'].shape[0])
        #import ipdb; ipdb.set_trace()

        loss, _, _ = sliced_score_estimation_vr(
            model, 
            batch['pixels'].to(device), 
            timesteps.to(device),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# %%
