import matplotlib.pyplot as plt
import numpy as np

# Function to create a curve - this is a placeholder, actual function should be based on real data or equations
def curve(x, type):
    if type == 'upper':
        return 1 - 0.5 * np.sqrt(x)
    elif type == 'lower':
        return 0.5 * np.sqrt(x)
    else:
        return 0.5 * np.ones_like(x)

# Create a figure and axis
fig, ax = plt.subplots()

# Define the range for the x-axis
x = np.linspace(0, 1, 100)

# Plot the curves
ax.plot(x, curve(x, 'upper'), 'blue')  # Upper boundary curve
ax.plot(x, curve(x, 'lower'), 'blue')  # Lower boundary curve

# Fill the areas between the curves
ax.fill_between(x, curve(x, 'upper'), 1, color='skyblue', alpha=0.5, label='Area C')
ax.fill_between(x, curve(x, 'lower'), curve(x, 'upper'), color='lightgreen', alpha=0.5, label='Area B')
ax.fill_between(x, 0, curve(x, 'lower'), color='sandybrown', alpha=0.5, label='Area A & D')

# Annotate areas
ax.text(0.1, 0.6, 'B', fontsize=12, verticalalignment='center', horizontalalignment='center')
ax.text(0.1, 0.9, 'C', fontsize=12, verticalalignment='center', horizontalalignment='center')
ax.text(0.8, 0.2, 'D', fontsize=12, verticalalignment='center', horizontalalignment='center')

# Set labels for the axes
ax.set_xlabel('X axis')
ax.set_ylabel('S axis')

# Set the axis limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Display the plot
plt.show()
