import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from neuron_model import NeuronModel

# Simulating neuron dynamics
# Fig : Saddle node neuron model
# Basic integrator model 
# Takes in currrent until it reaches a threshold and then it spikes and resets
# Neuronal firing equations
# Stable, unstable and saddle nodes
# Limit cycle and equilibria points

neuron = NeuronModel('saddle_node')    
dt = 0.01
T = 50
t = np.arange(0, T, dt)
amp = 5.8
neuron.dt = dt
# Creating different input currents
I_step = neuron.create_step_current(t, 1, 40, 0, 5)
I_ramp = neuron.create_ramp_current(t, 1, 10, 0, amp)
pulse_times = np.arange(0, 50, 5)
I_pulse = neuron.create_pulse_train(t, pulse_times, 3.5, 0, amp)
I_ext = I_pulse
a = neuron.simulate(T, dt, [-70, 0], I_ext)    

# Equilibria and limit cycle
equilibria = neuron.find_equlibrium_points(0, [-90, 20])
limit_cycle = neuron.find_limit_cycle(1)

# --- Create Figure & Grid Layout ---
fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1])

plt.suptitle("Integrator", fontsize=18)
# --- Left Panel (Membrane Potential vs Time) ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_ylabel("Membrane Potential (mV)", fontsize=14)
ax1.set_xticklabels([])
# ax1.set_ylim([-80, 20])
# ax1.set_title("Membrane Potential vs Time")
ax1.grid(True, linestyle="--", alpha=0.6)
line0, = ax1.plot(a[0][:], a[1][:, 0][:], marker = 'o', markevery = [-1])


# --- Right Panel (Phase Space Plot) ---
ax2 = fig.add_subplot(gs[:, 1])
ax2.set_xlabel("Membrane Potential (mV)", fontsize=14)
ax2.set_ylabel("$K^+$ activation", fontsize=14)
ax2.set_title("Phase Plot")
# ax2.set_xlim([np.min(a[1][:, 0]) - 5, np.max(a[1][:, 0]) + 5])
# ax2.set_ylim([np.min(a[1][:, 1]) - 1, np.max(a[1][:, 1]) + 1])
ax2.grid(True, linestyle="--", alpha=0.6)
line1, = ax2.plot(a[1][:, 0], a[1][:, 1], marker = 'o', markevery = [-1], color = 'black', linewidth = 1)


# Equilibrium points
for eq in equilibria:
    ax2.scatter(eq['point'][0], eq['point'][1], label=eq['stability'], zorder=3)
ax2.plot(limit_cycle[0], limit_cycle[1], color='blue', linestyle=':', label='Limit Cycle', alpha = 0.1)
ax2.legend()

# --- Bottom Panel (Current vs Time) ---
ax3 = fig.add_subplot(gs[1, 0])  # Takes full width
ax3.set_xlabel("Time (ms)", fontsize=14)
ax3.set_ylabel("$I_{ext}$ ($uA/cm^2$)", fontsize=14)
# ax3.set_title("External Current vs Time")
ax3.grid(True, linestyle="--", alpha=0.6)
# line2, = ax3.plot(a[0], I_ext, marker = 'o', markevery = [-1])
line2, = ax3.plot(a[0], I_ext(t), marker = 'o', markevery = [-1])

# --- Animation Update Function ---
def update(frame):
    if frame % 100 == 0:
        print(f'Rendering Frame {frame}')   
    # Membrane potential vs time
    line0.set_data(a[0][:frame], a[1][:, 0][:frame]) 

    # Phase space (V vs n)
    line1.set_data(a[1][:, 0][:frame], a[1][:, 1][:frame])

    # External current vs time
    line2.set_data(a[0][:frame], I_ext(t[:frame]))
    # line2.set_data(a[0][:frame], I_ext[:frame])

    return line0, line1, line2

# Create animation
print("Creating animation...")
print("Total frames:", len(a[0]))
ani = animation.FuncAnimation(fig, update, frames=range(0, len(a[0]), 5), interval=0.0001, blit=True)
plt.tight_layout()
# plt.show()

ani.save('Neuronal-Dynamics/animations/integrator.mp4', writer='ffmpeg', fps=30)
print("Done saving as mp4.")