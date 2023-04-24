import torch
import score_model

device = torch.device("cuda")

model_opts = {
    'normalization_groups': 32,
    'in_channels': 3,
    'out_channels': 3,
    'channels': 256,
    'num_head_channels': 64,
    'num_residuals': 6,
    'channel_multiple_schedule': [1, 2, 3],
    'interior_attention': 1,
}

optimizer_params = {
    'lr': 1e-4,
}

def model_optimizer():
    model = score_model.Diffuser(**model_opts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)

    return model, optimizer
