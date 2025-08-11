import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate dummy data for two different files
# This part is for demonstration. You can skip this if you have your own data files.
print("Generating dummy data...")
episodes_count = 10000
time = np.arange(0, episodes_count)

# Data for the first plot (Success Rate)
success_rate_dummy = 0.5 * (1 + np.sin(time / 1000)) + np.random.rand(episodes_count) * 0.2
success_rate_dummy = np.clip(success_rate_dummy, 0, 1)
np.savetxt("data_success_rate.txt", success_rate_dummy)
print("Dummy success rate data saved to data_success_rate.txt")

# Data for the second plot (Reward)
reward_dummy = np.log(time + 1) * 10 + np.random.randn(episodes_count) * 2 + 50
reward_dummy[0] = 50 # ensure the first value is not -inf
np.savetxt("data_reward.txt", reward_dummy)
print("Dummy reward data saved to data_reward.txt")

# Step 2: Read data from both text files
print("Reading data from files...")
success_rate_data = np.loadtxt('data_success_rate.txt')
reward_data = np.loadtxt('data_reward.txt')
episodes = np.arange(1, len(success_rate_data) + 1)

# Step 3: Plot and save the first figure (Success Rate)
print("Generating Success Rate plot...")
# Use a style that is suitable for academic papers
plt.style.use('seaborn-v0_8-paper')

# Create the first figure
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Plot the success rate data
ax1.plot(episodes, success_rate_data, color='darkblue', linewidth=0.8)
ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Success Rate', fontsize=12)
ax1.set_title('Success Rate over Episodes', fontsize=14, fontweight='bold')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.set_ylim(0, 1.05)
ax1.set_xlim(0, len(episodes))
plt.tight_layout()

# Save the first figure
plt.savefig('success_rate_plot.png', dpi=300)
print("Plot saved to success_rate_plot.png")
plt.close(fig1) # Close the figure to free memory

# Step 4: Plot and save the second figure (Reward)
print("Generating Reward plot...")

# Create the second figure
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Plot the reward data
ax2.plot(episodes, reward_data, color='darkgreen', linewidth=0.8)
ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Reward', fontsize=12)
ax2.set_title('Reward over Episodes', fontsize=14, fontweight='bold')
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.set_xlim(0, len(episodes))
plt.tight_layout()

# Save the second figure
plt.savefig('reward_plot.png', dpi=300)
print("Plot saved to reward_plot.png")
plt.close(fig2) # Close the figure