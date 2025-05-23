
import numpy as np
import matplotlib.pyplot as plt

# Simulate synthetic daily customer purchases data from an online store
np.random.seed(1999)
true_mu = 150   # True average daily purchases
true_sigma = 20  # True daily variability in purchases
data = np.random.normal(true_mu, true_sigma, size=100)  # 100 days of purchases


prior_mu_mean = 140         
prior_mu_precision = 0.01   
prior_sigma_alpha = 3       
prior_sigma_beta = 200


posterior_mu_precision = prior_mu_precision + len(data) / true_sigma**2
posterior_mu_mean = (prior_mu_precision * prior_mu_mean + np.sum(data) / true_sigma**2) / posterior_mu_precision

posterior_sigma_alpha = prior_sigma_alpha + len(data) / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data))**2) / 2

# Sample from the posterior distributions
posterior_mu = np.random.normal(posterior_mu_mean, 1 / np.sqrt(posterior_mu_precision), size=10000)
posterior_sigma = np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins=30, density=True, color='skyblue', edgecolor='black')
plt.title('Posterior Distribution of Daily Customer Purchases (μ)')
plt.xlabel('Average Daily Purchases')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=30, density=True, color='lightgreen', edgecolor='black')
plt.title('Posterior Distribution of Daily Purchases Variability (σ)')
plt.xlabel('Standard Deviation of Daily Purchases')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

mean_mu = np.mean(posterior_mu)
std_mu = np.std(posterior_mu)
print("Estimated Mean of Daily Purchases (μ):", round(mean_mu, 2))
print("Standard Deviation of Estimated Mean:", round(std_mu, 2))

mean_sigma = np.mean(posterior_sigma)
std_sigma = np.std(posterior_sigma)
print("Estimated Std Dev of Daily Purchases (σ):", round(mean_sigma, 2))
print("Standard Deviation of Estimated Std Dev:", round(std_sigma, 2))
