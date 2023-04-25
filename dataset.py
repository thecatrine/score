# %%
from datasets import load_dataset

import numpy as np
import torch
import einops

dataset = load_dataset("cifar10")

def map_func(examples):
    #import ipdb; ipdb.set_trace()
    examples['pixels'] = []
    for ex in examples['img']:
        im = np.array(ex)
        tensor = torch.Tensor(im)
        normalized_tensor = tensor / 255.0
        examples['pixels'].append(einops.rearrange(normalized_tensor.unsqueeze(0), 'b x y c -> b c x y'))
        
    return examples

train_dataset = dataset['train'].map(map_func, batched=True).with_format('torch')
test_dataset = dataset['test'].map(map_func, batched=True).with_format('torch')
# %%
