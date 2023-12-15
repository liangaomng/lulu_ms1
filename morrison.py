import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the cylinder and fluid
rho = 1000  # Density of water in kg/m^3
diameter = 1.0  # Diameter of the cylinder in meters
radius = diameter / 2
Cd = 1.2  # Drag coefficient
Cm = 2.0  # Added mass coefficient
A = np.pi * radius**2  # Cross-sectional area of the cylinder
V0 = A  # Reference volume flow speed

# Define the fluid velocity and acceleration
# For simplicity, we assume a sinusoidal wave for the velocity potential
wave_frequency = 100.0  # Frequency of the passing wave in Hz
wave_amplitude = 1.0  # Amplitude of the wave in m/s
time = np.linspace(0, 2*np.pi, 100) / wave_frequency  # Two periods of wave
u_x = wave_amplitude * np.sin(wave_frequency * time)  # Wave velocity profile
du_x_dt = np.gradient(u_x, time)  # Time derivative of the velocity

# Compute damping force and inertia force using Morrison's equation
f_d = 0.5 * Cd * rho * A * np.abs(u_x) * u_x
f_i = (rho * V0 + Cm * rho * V0) * du_x_dt

# Plotting the forces over time
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(time, f_d, label='Damping Force')
plt.title('Damping Force over Time')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time, f_i, label='Inertia Force')
plt.title('Inertia Force over Time')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
