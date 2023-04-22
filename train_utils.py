import torch

def save_state(model, optimizer, file):
    arrs = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(arrs, file)


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
