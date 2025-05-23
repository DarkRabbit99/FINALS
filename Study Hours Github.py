import numpy as np
import matplotlib.pyplot as plt

np.random.seed(176)
true_mu = 5.5    # true average study hours per day
true_sigma = 1.5  # true variability in study hours
data = np.random.normal(true_mu, true_sigma, size=100) 

# Define prior hyperparameters for the mean study hours
prior_mu_mean = 6  
prior_mu_precision = 0.3  
prior_sigma_alpha = 2  
prior_sigma_beta = 1  


posterior_mu_precision = prior_mu_precision + len(data) / true_sigma**2
posterior_mu_mean = (prior_mu_precision * prior_mu_mean + np.sum(data)) / posterior_mu_precision

posterior_sigma_alpha = prior_sigma_alpha + len(data) / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data))**2) / 2


posterior_mu = np.random.normal(posterior_mu_mean, 1 / np.sqrt(posterior_mu_precision), size=10000)
posterior_sigma = np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000)


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins=30, density=True, color='skyblue', edgecolor='black')
plt.title('Posterior Distribution of Average Daily Study Hours (μ)')
plt.xlabel('Study Hours (hours)')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=30, density=True, color='lightgreen', edgecolor='black')
plt.title('Posterior Distribution of Study Hours Variability (σ)')
plt.xlabel('Standard Deviation (hours)')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Summary statistics
mean_mu = np.mean(posterior_mu)
std_mu = np.std(posterior_mu)
print("Estimated Mean Daily Study Hours:", mean_mu)
print("Standard Deviation of Estimated Mean:", std_mu)

mean_sigma = np.mean(posterior_sigma)
std_sigma = np.std(posterior_sigma)
print("Estimated Standard Deviation of Study Hours:", mean_sigma)
print("Standard Deviation of Estimated Sigma:", std_sigma)
