import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from neuron_model import NeuronModel

# Simulating neuron dynamics
# Takes two neurons and compares their dynamics
# Same input current, slightly different initial state 
# Drastically different dynamics

neuron = NeuronModel('saddle_node')
dt = 0.01
T = 20
t = np.arange(0, T, dt)

amp = 1
neuron.dt = dt

# Creating different input currents
I_step = neuron.create_step_current(t, 0.1, 20, 0, amp)
I_ramp = neuron.create_ramp_current(t, 1, 10, 0, amp)
pulse_times = [0, 5, 10, 15, 20, 25, 30]  
I_pulse = neuron.create_pulse_train(t, pulse_times, 3, 0, 50)
I_ext = I_step
# a = neuron.simulate(T, dt, [-35, 0], I_ext)
# b = neuron.simulate(T, dt, [-40, 0], I_ext)
a = neuron.simulate_with_perturbations(T, dt, [-40, 0], I_ext, perturbations=[(0, 0, 0.0)])
b = neuron.simulate_with_perturbations(T, dt, [-40, 0], I_ext, perturbations=[(0.01, +4, 0.0)])
# Equilibria and limit cycle
equilibria = neuron.find_equlibrium_points(amp, [-90, 20])
limit_cycle = neuron.find_limit_cycle(1)

# --- Create Figure & Grid Layout ---
fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1])

plt.suptitle("Same input, different internal states", fontsize=18)
# --- Left Panel (Membrane Potential vs Time) ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_ylabel("Membrane Potential (mV)", fontsize=14)
ax1.set_xticklabels([])
# ax1.set_ylim([-80, 20])
# ax1.set_title("Membrane Potential vs Time")
ax1.grid(True, linestyle="--", alpha=0.6)
line0, = ax1.plot(a[0][:], a[1][:, 0][:], marker = 'o', markevery = [-1], color = 'indigo', label = 'Neuron 1')
line0b, = ax1.plot(b[0][:], b[1][:, 0][:], marker = 'o', markevery = [-1], color = 'tomato', label = 'Neuron 2')


# --- Right Panel (Phase Space Plot) ---
ax2 = fig.add_subplot(gs[:, 1])
ax2.set_xlabel("Membrane Potential (mV)", fontsize=14)
ax2.set_ylabel("$K^+$ activation", fontsize=14)
ax2.set_title("Phase Plot", fontsize=20)
# ax2.set_xlim([np.min(a[1][:, 0]) - 5, np.max(a[1][:, 0]) + 5])
ax2.set_ylim(-0.1, 0.8)
ax2.set_xlim(-80, 0)
ax2.grid(True, linestyle="--", alpha=0.6)
line1, = ax2.plot(a[1][:, 0], a[1][:, 1], marker = 'o', markevery = [-1], color = 'indigo', alpha = 0.6, label = 'Neuron 1')
line1b, = ax2.plot([], [], marker='o', markevery=[-1], color = 'tomato', alpha = 0.6, label = 'Neuron 2')   # Phase space for a2


# Equilibrium points
for eq in equilibria:
    ax2.scatter(eq['point'][0], eq['point'][1], label=eq['stability'], zorder=3, marker='X', s=100, edgecolor='black')
    # ax2.scatter(eq['point'][0], eq['point'][1], label=eq['stability'], zorder=3, marker='X', s=100, edgecolor='black')
# ax2.plot(limit_cycle[0], limit_cycle[1], color='blue', linestyle=':', label='Limit Cycle')
ax2.legend()

# --- Bottom Panel (Current vs Time) ---
ax3 = fig.add_subplot(gs[1, 0])  # Takes full width
ax3.set_xlabel("Time (ms)", fontsize=14)
ax3.set_ylabel("$I_{ext}$ ($uA/cm^2$)", fontsize=14)
# ax3.set_title("External Current vs Time")
ax3.grid(True, linestyle="--", alpha=0.6)
line2, = ax3.plot(a[0], I_ext, marker = 'o', markevery = [-1])
line2b, = ax3.plot(b[0], I_ext, marker = 'o', markevery = [-1], color = 'teal', alpha = 0.2, lw=1)
# line2, = ax3.plot(a[0], I_ext(t), marker = 'o', markevery = [-1])

# --- Animation Update Function ---
def update(frame):
    if frame % 100 == 0:
        print(f'Rendering Frame {frame}')   
    # Membrane potential vs time
    line0.set_data(a[0][:frame], a[1][:, 0][:frame])
    line0b.set_data(b[0][:frame], b[1][:, 0][:frame])

    # Phase space (V vs n)
    line1.set_data(a[1][:, 0][:frame], a[1][:, 1][:frame])
    line1b.set_data(b[1][:, 0][:frame], b[1][:, 1][:frame])

    # External current vs time
    # line2.set_data(a[0][:frame], I_ext(t[:frame]))
    line2.set_data(a[0][:frame], I_ext[:frame])
    line2b.set_data(b[0][:frame], I_ext[:frame])


    return line0, line1, line2, line0b, line1b, line2b

# Create animation
print("Creating animation...")
print("Total frames:", len(a[0]))
ani = animation.FuncAnimation(fig, update, frames=range(0, len(a[0]), 5), interval=0.0001, blit=True)
plt.tight_layout()
# plt.show()

ani.save('Neuronal-Dynamics/animations/two_neurons_internal.mp4', writer='ffmpeg', fps=60)
print("Done saving as mp4.")