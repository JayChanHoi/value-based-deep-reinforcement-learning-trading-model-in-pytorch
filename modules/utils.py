import torch
import numpy as np
import os

def policy(model, state, epsilon, device):
    # ----------------------------- epsilon greedy policy ----------------------------------------------
    # random policy
    action  = np.random.randint(3)

    # greedy polcy
    if np.random.rand() > epsilon:
        action = model(torch.Tensor(np.array(state, dtype=np.float32)).view(1, -1).to(device))
        action = np.argmax(action.data)

    return action

def shuffle_tensor(size, device):
    shuffle_index = torch.randperm(size).to(device)

    return shuffle_index

def resume(model, cuda, resume_checkpoint):
    print('=> loading checkpoint : "{}"'.format(resume_checkpoint))
    model_dict = model.state_dict()
    if not cuda:
        checkpoint = torch.load(resume_checkpoint, map_location=lambda storage, loc: storage)['state_dict']
        checkpoint = {k: v for k,v in checkpoint.items() if k in model_dict}
    else:
        checkpoint = torch.load(resume_checkpoint)['state_dict']
        checkpoint = {k: v for k,v in checkpoint.items() if k in model_dict}
    model_dict.update(checkpoint)
    model.load_state_dict(model_dict)

    return


if __name__ == "__main__":
    shuffled_tensor = shuffle_tensor(32, 'cpu')
    print(shuffled_tensor)

