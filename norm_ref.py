import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define the parameters for the normal distribution
mean = 0
std_dev = 1

# Generate values from -4 to 4
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)

# Calculate the normal distribution for these values
y = norm.pdf(x, mean, std_dev)

# Plot the normal distribution
plt.plot(x, y, label='Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Normal Distribution (mean=0, std_dev=1)')
plt.legend()
plt.grid(True)
plt.show()
