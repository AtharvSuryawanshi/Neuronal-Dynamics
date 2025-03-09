import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from neuron_model import NeuronModel

# Simulating neuron dynamics
sub_hop = NeuronModel('subcritical_Hopf')
dt = 0.01   
T = 100
t = np.arange(0, T, dt)

# Creating different input currents
I_step = sub_hop.create_step_current(t, 1, 80, 0, 5)
I_ramp = sub_hop.create_ramp_current(t, 1, 40, 0, 10)
pulse_times = [0, 6, 10, 15, 20, 30, 31]  
I_pulse = sub_hop.create_pulse_train(t, pulse_times, 3, 0, 50)
I_ext = I_pulse
a = sub_hop.simulate(T, dt, [-70, 0], I_ext)    

# Equilibria and limit cycle
equilibria = sub_hop.find_equlibrium_points(0, [-90, 20] )
limit_cycle = sub_hop.find_limit_cycle(1)

# --- Create Figure & Grid Layout ---
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1])

# --- Left Panel (Membrane Potential vs Time) ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_ylabel("Membrane Potential (mV)")
# ax1.set_xticks([])
# ax1.set_ylim([-80, 20])
ax1.set_title("Membrane Potential vs Time")
ax1.grid(True, linestyle="--", alpha=0.6)
line0, = ax1.plot(a[0][:], a[1][:, 0][:], marker = 'o', markevery = [-1])


# --- Right Panel (Phase Space Plot) ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlabel("Membrane Potential (mV)")
ax2.set_ylabel("n")
ax2.set_title("Phase Plot")
# ax2.set_xlim([np.min(a[1][:, 0]) - 5, np.max(a[1][:, 0]) + 5])
# ax2.set_ylim([np.min(a[1][:, 1]) - 1, np.max(a[1][:, 1]) + 1])
ax2.grid(True, linestyle="--", alpha=0.6)
line1, = ax2.plot(a[1][:, 0], a[1][:, 1], marker = 'o', markevery = [-1])


# Equilibrium points
for eq in equilibria:
    ax2.scatter(eq['point'][0], eq['point'][1], label=eq['stability'], zorder=3)
ax2.plot(limit_cycle[0], limit_cycle[1], color='blue', linestyle=':', label='Limit Cycle')
ax2.legend()

# --- Bottom Panel (Current vs Time) ---
ax3 = fig.add_subplot(gs[1, :])  # Takes full width
ax3.set_xlabel("Time (ms)")
ax3.set_ylabel("I_ext ($uA/cm^2$)")
ax3.set_title("External Current vs Time")
ax3.grid(True, linestyle="--", alpha=0.6)
line2, = ax3.plot(a[0], I_ext(t), marker = 'o', markevery = [-1])

# --- Animation Update Function ---
def update(frame):
    # Membrane potential vs time
    line0.set_data(a[0][:frame], a[1][:, 0][:frame]) 

    # Phase space (V vs n)
    line1.set_data(a[1][:, 0][:frame], a[1][:, 1][:frame])

    # External current vs time
    line2.set_data(a[0][:frame], I_ext(t[:frame]))

    return line0, line1, line2

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(a[0]), interval=0.0001, blit=True)

plt.tight_layout()
plt.show()