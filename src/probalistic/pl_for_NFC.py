"""
Neural ODEs: Making Normalizing Flows Continuous
In the Normalizing Flow based on affine coupling layers that we had considered in the previous section, there are a total of 6 layers and we could observe how the initial Gaussian distribution is transformed step-by-step to the target distribution.

A big limitation of this type of normalizing flow is that the architecture needs to be invertible. Most efficient architectures in deep learning are not invertible.

Our next goal is to get rid of this limitation so that we can use an arbitrary architecture. At the same time, this will address another shortcoming. Right now, the number of layers is fixed. Is it possible that instead of using a fixed number of layers, we can make the number of layers variable? If we imagine the transformation of the sampling distribution to the target distribution to be a continuous process instead of a sequence of invertible mappings, we can introduce an artificial “time” 
. Our normalizing flow should transform the sampling distribution smoothly from time 
 to the target distribution at time 
.

The way in which we can achieve this are Neural ODEs[CRBD19]. They’re especially interesting in the context of physics simulations. Mathematically speaking, they replace the mapping 
 with a learned velocity predictor 
. This means

 
Then the sequence of 
 steps that made up 
 can be replaced by integrating the velocity, which gives a simple ODE integration. The continuous time axis is introduced, with 
 starting at a normal Gaussian distribution, to 
 for the target distribution. The ODE is solved along this timeline, e.g. to transform the base distribution 
 into the target 
 by querying 
 for a velocity at each time point along the way. Even better, for an ODE solve there’s an analytic formulation for the gradient of the backpropagation path. This is a neat example of a differentiable physics solver (the ODE solve), providing an efficient way to compute a gradient, and aligns with the topics discussed in Scale-Invariance and Inversion.

The change in probabilities over time can also be computed conveniently via the trace of the learned function:

 
 
Compared to the Normalizing Flows above, an important difference in the NeuralODE picture is that now we have a single function 
 that is repeatedly evaluated at different points in the time interval 
. This might seem like a trivial change at first, but it’s a crucial step towards more powerful probabilistic models such as diffusion models. It turns out to be important that we can re-use a learned function, instead of having to manually construct many layers with large numbers of trainable parameters.

Building a Continuous Normalizing Flow
In the next cell’s we’ll use the Free-form Jacobian of Reversible Dynamics (FFJORD) architecture (from here) to implement a continuous normalizing flow.
"""

import torch.nn as nn

def kernel_init_fn():
    return nn.init.xavier_uniform_

def bias_init_fn():
    return nn.init.zeros_

class ConcatSquash(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.out_size = out_size

        self.lin1 = nn.Linear(in_size, out_size)
        self.lin2 = nn.Linear(1, out_size)
        self.lin3 = nn.Linear(1, out_size, bias=False)

        kernel_init = kernel_init_fn()
        kernel_init(self.lin1.weight)
        kernel_init(self.lin2.weight)
        kernel_init(self.lin3.weight)

        bias_init = bias_init_fn()
        bias_init(self.lin1.bias)
        bias_init(self.lin2.bias)

    def forward(self, t, y):
        if t.dim() == 0:
            t = t.view(1, 1)
        elif t.dim() == 1:
            t = t.view(-1, 1)

        return self.lin1(y) * torch.sigmoid(self.lin2(t)) + self.lin3(t)

class FFJORD(nn.Module):
    def __init__(self, data_size, width_size, depth):
        super().__init__()
        self.data_size = data_size
        self.width_size = width_size
        self.depth = depth

        layers = []

        if self.depth == 0:
            layers.append(ConcatSquash(in_size=data_size, out_size=self.data_size))
        else:
            layers.append(ConcatSquash(in_size=data_size, out_size=self.width_size))
            for _ in range(self.depth - 1):
                layers.append(ConcatSquash(in_size=width_size, out_size=self.width_size))
            layers.append(ConcatSquash(in_size=width_size, out_size=self.data_size))

        self.layers = nn.ModuleList(layers)

    def forward(self, t, y):
        for layer in self.layers[:-1]:
            y = layer(t, y)
            y = torch.tanh(y)
        y = self.layers[-1](t, y)
        return y

"""
The Continuous Normalizing Flow Network
For ODE integration, we’ll make use of the differentiable ODE solvers from the torchdiffeq package.
"""
try:
    import google.colab  # only to ensure that we are inside colab
    %pip install --quiet torchdiffeq
except ImportError:
    print("This notebook is running locally, please install torchdiffeq manually.")

"""
The ContinuousNormalizingFlow class implements the basic functionality to integrate the neural velocity estimator via odeint in a differentiable manner. The latter is important to allow for backpropagating the gradients from the loss (and the output of the integration step) back to the weights of the FFJORD network.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint

class CNFVelocityFn(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=8):
        super().__init__()

        self.net = FFJORD(data_size=input_dim, width_size=hidden_dim, depth=num_layers)

    def forward(self, t, combined_state):
        y, ldj = combined_state

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)

            t = torch.unsqueeze(t.repeat(y.shape[0]), 1)

            velocity = self.net(t, y)

            divergence = 0.0
            for i in range(y.shape[1]):
                divergence += torch.autograd.grad(velocity[:, i].sum(), y, create_graph=True)[0][:, i]

        return velocity, divergence.view(velocity.shape[0], 1)

class ContinuousNormalizingFlow(nn.Module):

    def __init__(self, input_dim, hidden_dim,
                 time_0=0.0, time_T=1.0):

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_0 = time_0
        self.time_T = time_T

        self.velocity_fn = CNFVelocityFn(input_dim=input_dim, hidden_dim=hidden_dim)

    def solveODE(self, x, t):

        batch_size, dim = x.shape
        assert dim == self.input_dim, "Input dimension mismatch!"

        y0 = x
        ldj0 = torch.zeros(batch_size, device=x.device)

        combined_state = (y0, ldj0)

        result = odeint(self.velocity_fn, combined_state, t,
                        method='dopri5',
			            atol=[1e-5, 1e-5],
			            rtol=[1e-5, 1e-5],
                        )

        final_y, final_ldj = result

        return final_y[-1], final_ldj[-1]

    def forward(self, x):

        t = torch.tensor([self.time_0, self.time_T], device=x.device)

        return self.solveODE(x, t)

    def inverse(self, x):

        t = torch.tensor([self.time_T, self.time_0], device=x.device)

        return self.solveODE(x, t)

"""
Training
The training step can re-use the train_model function from above, as all basic modalities (data format, loss, etc.) stay the same. We’re only replacing the “discrete” step-by-step transformation with the continuously integrated version.
"""

samples = generate_2d_gaussian_mixture(5000, gm) # use fewer samples because training takes longer
samples = shuffle(samples.numpy())
dataset = TensorDataset(torch.tensor(samples, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

input_dim = 2
hidden_dim = 128
time_0 = 0.0
time_T = 1.0

cnf_model = ContinuousNormalizingFlow(input_dim=input_dim, hidden_dim=hidden_dim, time_0=time_0, time_T=time_T)

learning_rate = 2e-4
optimizer = torch.optim.Adam(cnf_model.parameters(), lr=learning_rate)

num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
losses = train_model(cnf_model, dataloader, optimizer, num_epochs=num_epochs, device=device)

