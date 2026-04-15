"""
Visualization of the Continuous Normalizing Flow
Now we can repeat the same steps as before to visualize what this second network has learned.
"""

visualize_training_results(cnf_model.to(device), gm, 100, model_desc='CNF')


def visualize_progression_with_time_and_likelihoods(model, grid_size=100, num_timepoints=6, num_samples=1000):
    model.eval()

    timepoints = torch.linspace(0.0, 1.0, num_timepoints)
    fig, axes = plt.subplots(2, num_timepoints, figsize=(20, 8))

    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    points_tensor = torch.tensor(points, dtype=torch.float32).to(device)

    z = torch.randn(num_samples, 2).to(device)

    c = get_angle_colors(z.cpu().numpy())

    eps = 1e-5

    for i, t in enumerate(timepoints):
        t_tensor = torch.tensor([0.0, t+eps]).to(device)
        t_tensor_inv = torch.tensor([1.0, 1.0-t-eps]).to(device)

        with torch.no_grad():

            samples, _ = model.solveODE(z, t_tensor_inv)
            samples = samples.cpu().numpy()

            z_t, ldj = model.solveODE(points_tensor, t_tensor)

            prior = (-0.5 * z_t ** 2).sum(-1) - 0.5 * torch.log(torch.tensor(2.0 * torch.pi))

            likelihoods = torch.exp(prior + ldj).cpu().numpy().reshape(grid_size, grid_size)

        scatter_ax = axes[0, i]
        scatter_ax.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.7, c=c)
        scatter_ax.set_title(f"t = {t:.1f}")
        scatter_ax.set_xlim(-5, 5)
        scatter_ax.set_ylim(-5, 5)
        scatter_ax.set_xlabel("X" if i == 0 else "")
        scatter_ax.set_ylabel("Y" if i == 0 else "")

        likelihood_ax = axes[1, i]
        contour = likelihood_ax.contourf(X, Y, likelihoods, levels=50, cmap="viridis")
        likelihood_ax.set_xlim(-5, 5)
        likelihood_ax.set_ylim(-5, 5)
        likelihood_ax.set_xlabel("X")
        likelihood_ax.set_ylabel("Y" if i == 0 else "")

    plt.tight_layout()
    plt.show()

visualize_progression_with_time_and_likelihoods(
    cnf_model,
    grid_size=100,
    num_timepoints=6,
    num_samples=1000
)

"""
The figure shows how the Gaussian sampling distribution is transformed to the target distribution much more smoothly using the continuous-time normalizing flow than the implementation with discrete layers.

Summary of Normalizing Flows
This is a great result. Using our knowledge about ODE solving, we can choose basically any neural network architecture for the velocity. We can trade off speed against accuracy when solving the ODE by choosing different solvers and step sizes depending on our current computational budget.

However, there are also disadvantages with this approach. In order to train our continuous normalizing flow, we need to solve the entire ODE transporting the samples from our target distribution from 
 until 
 to the Gaussian distribution to evaluate their likelihoods with high accuracy. This requires a large number of network evaluations and is computationally expensive. As such, it is difficult to scale neural ODEs to high dimensional data and large neural networks.
"""
