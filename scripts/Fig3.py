import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from neuron_model import NeuronModel

# Simulating neuron dynamics
# Shows the dynamics phase space for saddle node neuron model

neuron = NeuronModel('saddle_node')
dt = 0.01
T = 20
t = np.arange(0, T, dt)

amp = 0
# Creating different input currents
I_step = neuron.create_step_current(t, 0.1, 80, 0, amp)
I_ramp = neuron.create_ramp_current(t, 1, 10, 0, amp)
pulse_times = [0, 5, 10, 15, 20, 25, 30]  
I_pulse = neuron.create_pulse_train(t, pulse_times, 3, 0, 50)
I_ext = I_step

num_trajectories = 200
initial_conditions = np.random.uniform(low=[-80, 0], high=[50, 1], size=(num_trajectories, 2))

a1 = neuron.simulate(T, dt, [-67, 0], I_ext)
trajectories = []
for init in initial_conditions:
    traj = neuron.simulate(T, dt, init, I_ext)
    trajectories.append(traj)

# Equilibria and limit cycle
equilibria = neuron.find_equlibrium_points(amp, [-90, 20])
limit_cycle = neuron.find_limit_cycle(amp)
# separatrix = neuron.find_separatrix(amp, [-90, 0])

# --- Create Figure & Grid Layout ---
fig, ax2 = plt.subplots(1,1, figsize=(10, 8))
# --- (Phase Space Plot) ---
ax2.set_xlabel("Membrane Potential (mV)", fontsize=14)  
ax2.set_ylabel("$K^+$ activation", fontsize=14)
ax2.set_title("Phase Plot", fontsize=20)
# ax2.set_ylim(-0.1, 1)
# ax2.set_xlim(-100, -20)
ax2.grid(True, linestyle="--", alpha=0.1)
line1, = ax2.plot(a1[1][:, 0], a1[1][:, 1], marker = 'o', markevery = [-1], color = 'black', linewidth = 0)
# line1b, = ax2.plot([], [], marker='o', markevery=[-1], color='green')   # Phase space for a2
# Before animation: Create line handles for each trajectory
lines = []
for traj in trajectories:
    # Plot an empty line initially; will update it during animation
    line, = ax2.plot(traj[1][:,0], traj[1][:, 1], markevery = [-1], color = 'teal', alpha = 0.2, lw=1, marker = '.')
    lines.append(line)

# Equilibrium points
for eq in equilibria:
    ax2.scatter(eq['point'][0], eq['point'][1], label=eq['stability'], zorder=3, marker='X', s=100, edgecolor='black')
    # ax2.scatter(eq['point'][0], eq['point'][1], label=eq['stability'], zorder=3, marker='X', s=100, edgecolor='black')
# ax2.plot(limit_cycle[0], limit_cycle[1], color='indigo', label='Limit Cycle', alpha = 1, marker='')
V_vals = np.linspace(-80, 20, 100)
# ax2.plot(separatrix[0], separatrix[1], color='crimson', linestyle='--', label='Separatrix', alpha = 1)
ax2.plot(V_vals, neuron.V_nullcline(V_vals, amp), color='red', linestyle='--', label='V Nullcline', alpha = 0.5)
ax2.plot(V_vals, neuron.n_nullcline(V_vals), color='green', linestyle='--', label='n Nullcline', alpha = 0.5)
ax2.legend()
ax2.legend()

# --- Animation Update Function ---
def update(frame):
    if frame % 100 == 0:
        print(f'Rendering Frame {frame}')   
    # line1.set_data(a1[1][:, 0][:frame], a1[1][:, 1][:frame])
    for i, traj in enumerate(trajectories):
        lines[i].set_data(traj[1][:, 0][:frame], traj[1][:, 1][:frame])
    return lines

# Create animation
print("Creating animation...")
print("Total frames:", len(a1[0]))
ani = animation.FuncAnimation(fig, update, frames=range(0, len(trajectories[0][1][:,0]), 5), interval=0.0001, blit=True)
plt.tight_layout()
plt.show()

# ani.save('Neuronal-Dynamics/animations/phase_trajectories.mp4', writer='ffmpeg', fps=60)
# print("Done saving as mp4.")