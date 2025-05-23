import numpy as np
import matplotlib.pyplot as plt


np.random.seed(2025)

# User inputs
try:
    before_customers = int(input("Number of customers who visited before the sale: "))
    before_buyers = int(input("Number of purchases before the sale: "))
    after_customers = int(input("Number of customers who visited during the sale: "))
    after_buyers = int(input("Number of purchases during the sale: "))
except ValueError:
    print("Invalid input. Using default values.")
    before_customers = 100
    before_buyers = 20
    after_customers = 120
    after_buyers = 50


before_rate = before_buyers / before_customers
after_rate = after_buyers / after_customers
print(f"\nConversion Rate Before Sale: {before_rate:.2%}")
print(f"Conversion Rate During Sale: {after_rate:.2%}")


alpha_prior = 1
beta_prior = 1


before_posterior = np.random.beta(alpha_prior + before_buyers,
                                  beta_prior + before_customers - before_buyers,
                                  size=10000)
after_posterior = np.random.beta(alpha_prior + after_buyers,
                                 beta_prior + after_customers - after_buyers,
                                 size=10000)


uplift = after_posterior - before_posterior


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(before_posterior, bins=50, alpha=0.7, label='Before Sale', color='skyblue', density=True)
plt.hist(after_posterior, bins=50, alpha=0.7, label='During Sale', color='lightgreen', density=True)
plt.axvline(before_rate, color='blue', linestyle='dashed')
plt.axvline(after_rate, color='green', linestyle='dashed')
plt.title('Posterior Conversion Rates')
plt.xlabel('Conversion Rate')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(uplift, bins=50, color='salmon', alpha=0.7, density=True)
plt.axvline(np.mean(uplift), color='red', linestyle='dashed')
plt.title('Estimated Conversion Rate Uplift Due to Sale')
plt.xlabel('Uplift')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Summary
mean_uplift = np.mean(uplift)
cred_int = np.percentile(uplift, [2.5, 97.5])
prob_positive_uplift = np.mean(uplift > 0)

print(f"\nEstimated Average Uplift: {mean_uplift:.2%}")
print(f"95% Credible Interval: [{cred_int[0]:.2%}, {cred_int[1]:.2%}]")
print(f"Probability that Sale Increased Conversion Rate: {prob_positive_uplift:.2%}")
