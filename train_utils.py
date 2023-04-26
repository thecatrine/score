import torch
import torchvision

def save_state(epoch, test_loss, model, optimizer, scheduler, file):
    arrs = {
        'epoch': epoch,
        'best_test_loss': test_loss,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
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


def mnist_rescale(f):
    f = f * 255
    return torchvision.transforms.ToPILImage()(f)


def scheduler_function(max_steps, warmup_frac, step):
    frac = step / max_steps

    if frac < warmup_frac:
        return frac / warmup_frac
    else:
        return 1.0 - ((frac - warmup_frac) / (1.0 - warmup_frac))