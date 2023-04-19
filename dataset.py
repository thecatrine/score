from datasets import load_dataset

import numpy as np
import torch

dataset = load_dataset("mnist")

def map_func(examples):
    #import ipdb; ipdb.set_trace()
    examples['pixels'] = []
    for ex in examples['image']:
        im = np.array(ex)
        tensor = torch.Tensor(im)
        normalized_tensor = tensor / 255.0
        normalized_tensor += torch.randn(tensor.size())
        examples['pixels'].append(normalized_tensor.unsqueeze(0))
        
    return examples

train_dataset = dataset['train'].map(map_func, batched=True).with_format('torch')
test_dataset = dataset['test'].map(map_func, batched=True).with_format('torch')