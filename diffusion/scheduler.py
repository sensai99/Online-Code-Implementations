import torch

class Scheduler():
    def __init__(self, noise_steps = 1000, beta_start = 0.0001, beta_end = 0.02):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = self.get_noise_schedule()
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, 0)

    # Define the scheduler for the noise - linearly increasing from beta_start to beta_end
    def get_noise_schedule(self, type = 'linear'):
        if type == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        else:
            raise NotImplementedError(f'Noise schedule type {type} not implemented')
    
    # Apply the forward diffusion process
    def apply_forward_diffusion(self, x_0, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hats[t])[:, None, None, None]
        one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hats[t])[:, None, None, None]
        epsilon = torch.randn_like(x_0)
        return sqrt_alpha_hat * x_0 + one_minus_alpha_hat * epsilon, epsilon
    
    def sample_timesteps(self, num_timesteps):
        return torch.randint(0, self.noise_steps, (num_timesteps,))

