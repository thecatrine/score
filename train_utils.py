def save_state(model, optimizer, file):
    arrs = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(arrs, file)