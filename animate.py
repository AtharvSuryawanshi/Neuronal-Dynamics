import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import HH_model  

# Simulating neuron dynamics
model = HH_model('saddle_node')  # Choose the type of bifurcation
dt = 0.025
t = 100
I_ext = 1*np.arange(0, t, dt) 
a = model.simulate(t, dt, [-70, 0.3], I_ext)

# Extract data
time_points = a[0]
V_values = a[1][:, 0]  # Membrane potential

# Create figure
fig, ax = plt.subplots()
ax.set_xlim(time_points[0], time_points[-1])
ax.set_ylim(min(V_values) - 5, max(V_values) + 5)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Membrane Potential (mV)")
ax.set_title("Neuron Spiking Over Time")

# Line to update
line, = ax.plot([], [], lw=2)

def update(frame):
    line.set_data(time_points[:frame], V_values[:frame])
    return line,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(time_points), interval=0.1, blit=True)

plt.show()

# Extract phase portrait data
n_values = a[1][:, 1]

fig, ax = plt.subplots()
ax.set_xlim(min(V_values) - 5, max(V_values) + 5)
ax.set_ylim(min(n_values) - 0.1, max(n_values) + 0.1)
ax.set_xlabel("Membrane potential (mV)")
ax.set_ylabel("n (Gating variable)")
ax.set_title("Phase Portrait Animation")

# Line to update
phase_line, = ax.plot([], [], lw=2)

def update_phase(frame):
    phase_line.set_data(V_values[:frame], n_values[:frame])
    return phase_line,

# Create animation
phase_ani = animation.FuncAnimation(fig, update_phase, frames=len(V_values), interval=1, blit=True)
plt.show()
