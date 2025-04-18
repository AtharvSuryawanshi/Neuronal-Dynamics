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

neuron = NeuronModel('supercritical_Hopf')
dt = 0.01
T = 20
t = np.arange(0, T, dt)

# Creating different input currents
I_step = neuron.create_step_current(t, 0.1, 80, 0, 0)
I_ramp = neuron.create_ramp_current(t, 1, 10, 0, 5)
pulse_times = [0, 5, 10, 15, 20, 25, 30]  
I_pulse = neuron.create_pulse_train(t, pulse_times, 3, 0, 50)
I_ext = I_step

num_trajectories = 200
initial_conditions = np.random.uniform(low=[-80, 0], high=[50, 0.6], size=(num_trajectories, 2))

a1 = neuron.simulate(T, dt, [-67, 0], I_ext)
trajectories = []
for init in initial_conditions:
    traj = neuron.simulate(T, dt, init, I_ext)
    trajectories.append(traj)

# Equilibria and limit cycle
equilibria = neuron.find_equlibrium_points(0, [-90, 20])
limit_cycle = neuron.find_limit_cycle(1)

# --- Create Figure & Grid Layout ---
fig, ax2 = plt.subplots(1,1, figsize=(12, 8))
# --- (Phase Space Plot) ---
ax2.set_xlabel("Membrane Potential (mV)")
ax2.set_ylabel("n")
ax2.set_title("Phase Plot")
# ax2.set_ylim(-0.1, 1)
# ax2.set_xlim(-100, -20)
ax2.grid(True, linestyle="--", alpha=0.6)
line1, = ax2.plot(a1[1][:, 0], a1[1][:, 1], marker = 'o', markevery = [-1], color = 'black', linewidth = 0)
# line1b, = ax2.plot([], [], marker='o', markevery=[-1], color='green')   # Phase space for a2
# Before animation: Create line handles for each trajectory
lines = []
for traj in trajectories:
    # Plot an empty line initially; will update it during animation
    line, = ax2.plot(traj[1][:,0], traj[1][:, 1], markevery = [-1], color = 'teal', alpha = 0.2, lw=1)
    lines.append(line)

# Equilibrium points
for eq in equilibria:
    ax2.scatter(eq['point'][0], eq['point'][1], label=eq['stability'], zorder=3, marker='X', s=100, edgecolor='black')
    # ax2.scatter(eq['point'][0], eq['point'][1], label=eq['stability'], zorder=3, marker='X', s=100, edgecolor='black')
# ax2.plot(limit_cycle[0], limit_cycle[1], color='blue', linestyle=':', label='Limit Cycle')
ax2.legend()

# --- Animation Update Function ---
def update(frame):
    # line1.set_data(a1[1][:, 0][:frame], a1[1][:, 1][:frame])
    for i, traj in enumerate(trajectories):
        lines[i].set_data(traj[1][:, 0][:frame], traj[1][:, 1][:frame])
    return lines



# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(a1[1]), interval=0.1, blit=True)

plt.tight_layout()
plt.show()

# from matplotlib.animation import FFMpegWriter

# writer = FFMpegWriter(fps=30, metadata=dict(artist='YourName'), bitrate=1800)

# ani.save("animation.mp4", writer=writer)
