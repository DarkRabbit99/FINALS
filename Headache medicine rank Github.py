

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1999)

# Simulated pain reduction data (larger is better)
medicine_data = {
    'Paracetamol': np.random.normal(loc=4.5, scale=1.2, size=50),
    'Ibuprofen': np.random.normal(loc=5.5, scale=1.5, size=50),
    'Naproxen': np.random.normal(loc=5.0, scale=1.0, size=50),
}

# Ask user if they currently have a headache
has_headache = input("Do you currently have a headache? (yes/no): ").strip().lower()
if has_headache != 'yes':
    print("You do not have a headache right now. Proceeding with simulated data.")
else:
    print("Rate your pain before taking the medicine (0 to 10):")
    try:
        pre_pain = float(input("Pain before: ").strip())
        while not (0 <= pre_pain <= 10):
            pre_pain = float(input("Invalid input. Enter a number from 0 to 10: ").strip())
    except ValueError:
        pre_pain = 6.0
        print("Invalid input. Defaulting to 6.0")

    print("Choose one of the following medicines to take:")
    print("1. Paracetamol\n2. Ibuprofen\n3. Naproxen")
    med_choice = input("Enter 1, 2, or 3: ").strip()
    med_map = {'1': 'Paracetamol', '2': 'Ibuprofen', '3': 'Naproxen'}
    while med_choice not in med_map:
        med_choice = input("Invalid input. Choose 1, 2, or 3: ").strip()
    chosen_med = med_map[med_choice]

    try:
        post_pain = float(input("Pain after taking the medicine (0 to 10): ").strip())
        while not (0 <= post_pain <= 10):
            post_pain = float(input("Invalid input. Enter a number from 0 to 10: ").strip())
    except ValueError:
        post_pain = 2.0
        print("Invalid input. Defaulting to 2.0")

    reduction = max(0, pre_pain - post_pain)
    medicine_data[chosen_med] = np.append(medicine_data[chosen_med], reduction)
    print(f"Recorded: Your pain reduction ({reduction}) added to {chosen_med}.")


posterior_samples = {}
for med, data in medicine_data.items():
    n = len(data)
    sample_mean = np.mean(data)
    sample_var = np.var(data)

    
    prior_mu_mean = 5.0
    prior_mu_precision = 0.05
    prior_sigma_alpha = 3
    prior_sigma_beta = 2

    
    post_mu_precision = prior_mu_precision + n / sample_var
    post_mu_mean = (prior_mu_precision * prior_mu_mean + n * sample_mean / sample_var) / post_mu_precision
    post_sigma_alpha = prior_sigma_alpha + n / 2
    post_sigma_beta = prior_sigma_beta + np.sum((data - sample_mean) ** 2) / 2

    
    posterior_mu = np.random.normal(post_mu_mean, 1 / np.sqrt(post_mu_precision), size=5000)
    posterior_samples[med] = posterior_mu

# Visualization
colors = {'Paracetamol': 'skyblue', 'Ibuprofen': 'salmon', 'Naproxen': 'lightgreen'}

plt.figure(figsize=(10, 6))
for med, samples in posterior_samples.items():
    plt.hist(samples, bins=40, density=True, alpha=0.6, label=med, color=colors[med])
plt.title("Posterior Distribution of Headache Medicine Effectiveness")
plt.xlabel("Estimated Mean Pain Reduction")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Summary Stats
print("\n--- Summary of Estimated Effectiveness ---")
for med, samples in posterior_samples.items():
    print(f"{med}: Mean = {np.mean(samples):.2f}, Std Dev = {np.std(samples):.2f}")
