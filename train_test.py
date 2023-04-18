# %% 
import torch
import torch.nn.functional as F

import plotly.express as px
import plotly.figure_factory as ff

import numpy as np
import matplotlib.pyplot as plt

import tqdm
# %%

class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        
        self.fc1 = torch.nn.Linear(2, 10)
        self.fc2 = torch.nn.Linear(10, 10)
        self.fc3 = torch.nn.Linear(10, 2)

    def forward(self, inp):
        x = self.fc1(inp)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)

        return x


def testfunc(x, y):
    return torch.exp(-x**2 - y**2)

def calc(m, x, y):
    res = m(torch.Tensor([[x, y]]))
    return res

def sliced_score_estimation_vr(score_net, samples, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)

    grad1 = score_net(dup_samples)
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    grad2 = torch.autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()

def train():
    model = TestModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    m = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

    for i in range(10000):
        optimizer.zero_grad()
        

        x = m.sample((1000,))
        x.requires_grad_(True)

        # Calculate h(x)
        out = model(x)

        # Random +- 1 vectors in R^n
        v = torch.randint(0, 3, (1000, 2)).float() - 1
        #import ipdb; ipdb.set_trace()
        term_before_grad = (v * out).sum(1)

        # Need grad wrt x at x
        g = torch.autograd.grad(term_before_grad.sum(), x, create_graph=True)[0]



        out = model(x)
        l22 =((out * out).sum(1))
        J = 0.5 * l22

        J = J + (g * v).sum(1)
        
        loss = J.mean()

        #loss, _, _ = sliced_score_estimation_vr(model, x)

        loss.backward()
        optimizer.step()

        entropy_loss = 0

        #import ipdb; ipdb.set_trace()
        print(loss)

    return model
# %%

m = train()


# %%
foo = []

for x in range(-10, 10):
    for y in range(-10, 10):
        xx = x / 5
        yy = y / 5
        foo.append([xx, yy])

z = m(torch.Tensor(foo)).detach().numpy()
foo = np.array(foo)

plt.quiver(foo[:, 0], foo[:, 1], z[:, 0], z[:, 1])
# %%

m = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
xys = m.sample((1000,))
px.scatter(x=xys[:, 0], y=xys[:, 1])
# %%

# sampling from distribution

def langevin(m, x0, dt, steps=100):
    dist = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)*dt)

    x = x0
    for i in range(steps):
        bt = dist.sample()
        dx = m(x)*dt + 1.414*bt

        #import ipdb; ipdb.set_trace()

        x = x + dx

    return x
# %%

samples = []
for i in tqdm.tqdm(range(1000)):
    samples.append(langevin(m, torch.zeros(2), 0.01, 100).detach().numpy())
# %%
