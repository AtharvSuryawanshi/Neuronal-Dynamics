import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from neuron_model import NeuronModel

# Simulating neuron dynamics
# Fig: V and n nullclines and their change with time,also limit cycle


neuron = NeuronModel('supercritical_Hopf')  
dt = 0.01
T = 12
t = np.arange(0, T, dt)

# Creating different input currents
I_step = neuron.create_step_current(t, 1, 80, 0, 0)
I_ramp = neuron.create_ramp_current(t, 1, 10, -5, 30)
# I_ramp = neuron.create_ramp_current(t, 1, 10, 20, 70)
pulse_times = [0, 6, 10, 15, 20, 30, 31]  
I_pulse = neuron.create_pulse_train(t, pulse_times, 3, 0, 50)
I_ext = I_ramp
# a = neuron.simulate(T, dt, [-70, 0], I_ext)    
neuron.dt = dt
a = neuron.simulate_with_perturbations(T, dt, [- 30, 0.4], I_ext, perturbations=[(6.2, -10, 0.0)])

# Equilibria and limit cycle
equilibria = neuron.find_equlibrium_points(0, [-90, 20] )
# Placeholder for dynamic equilibrium points
limit_cycle = neuron.find_limit_cycle(1)

# --- Create Figure & Grid Layout ---
fig = plt.figure(figsize=(7, 9))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

# # --- Left Panel (Membrane Potential vs Time) ---
# ax1 = fig.add_subplot(gs[0, 0])
# ax1.set_ylabel("Membrane Potential (mV)")
# # ax1.set_xticks([])
# # ax1.set_ylim([-80, 20])
# ax1.set_title("Membrane Potential vs Time")
# ax1.grid(True, linestyle="--", alpha=0.6)
# line0, = ax1.plot(a[0][:], a[1][:, 0][:], marker = 'o', markevery = [-1])


# --- Right Panel (Phase Space Plot) ---
ax2 = fig.add_subplot(gs[0, 0])
ax2.set_xlabel("Membrane Potential (mV)")
ax2.set_ylabel("n")
ax2.set_title("Phase Plot")
# ax2.set_xlim([np.min(a[1][:, 0]) - 5, np.max(a[1][:, 0]) + 5])
# ax2.set_ylim([np.min(a[1][:, 1]) - 1, np.max(a[1][:, 1]) + 1])
ax2.grid(True, linestyle="--", alpha=0.6)
# line1, = ax2.plot(a[1][:, 0], a[1][:, 1], marker = 'o', markevery = [-1])
# Create a Line2D object for dynamic V-nullcline
dynamic_V_nullcline, = ax2.plot([], [], color='orange', linestyle='--', label='V Nullcline')
dynamic_limit_cycle, = ax2.plot([], [], color='indigo', label='Limit Cycle')
# dynamic_equilibria_plot, = ax2.plot([], [], marker = 'o', label='Equilibria', zorder=3, lw = 0)  # 'ko' means black circles
# Persistent plot handles for each equilibrium type
stable_eq_plot, = ax2.plot([], [], 'bo', label='Stable', zorder=3)     # Blue circles
unstable_eq_plot, = ax2.plot([], [], 'ro', label='Unstable', zorder=3) # Red circles
saddle_eq_plot, = ax2.plot([], [], 'yo', label='Saddle', zorder=3)     # Yellow circles


# Equilibrium points
# for eq in equilibria:
#     ax2.scatter(eq['point'][0], eq['point'][1], label=eq['stability'], zorder=3)
# ax2.plot(limit_cycle[0], limit_cycle[1], color='indigo', linestyle='--', label='Limit Cycle', alpha = 0.5)
V_vals = np.linspace(-80, 20, 100)
# ax2.plot(V_vals, neuron.V_nullcline(V_vals, 0), color='red', linestyle='--', label='V Nullcline', alpha = 0.5)
ax2.plot(V_vals, neuron.n_nullcline(V_vals), color='green', linestyle='--', label='n Nullcline', alpha = 0.5)
ax2.legend()

# --- Bottom Panel (Current vs Time) ---
ax3 = fig.add_subplot(gs[1, 0])  # Takes full width
ax3.set_xlabel("Time (ms)")
ax3.set_ylabel("I_ext ($uA/cm^2$)")
ax3.set_title("External Current vs Time")
ax3.grid(True, linestyle="--", alpha=0.6)
line2, = ax3.plot(a[0], I_ext, marker = 'o', markevery = [-1])
# line2, = ax3.plot(a[0], I_ext(t), marker = 'o', markevery = [-1])
# To store previously computed limit cycle
prev_limit_cycle_data = {"x": [], "y": []}


# --- Animation Update Function ---
def update(frame):
    # External current vs time
    line2.set_data(a[0][:frame], I_ext[:frame])

    # Update dynamic V-nullcline
    current_I = I_ext[frame]
    V_vals = np.linspace(-80, 20, 100)
    V_ncline = neuron.V_nullcline(V_vals, current_I)
    dynamic_V_nullcline.set_data(V_vals, V_ncline)

    # Update equilibria
    dynamic_equilibria = neuron.find_equlibrium_points(current_I, [-90, 20])

    # Update limit cycle only every 5th frame
    if frame % 100 == 0:
        lc = neuron.find_limit_cycle(current_I)
        if lc is not None:
            prev_limit_cycle_data["x"], prev_limit_cycle_data["y"] = lc[0], lc[1]
        else:
            prev_limit_cycle_data["x"], prev_limit_cycle_data["y"] = [], []

    # Always update the plot from previous data (even if not newly computed)
    dynamic_limit_cycle.set_data(prev_limit_cycle_data["x"], prev_limit_cycle_data["y"])

    # Classify and plot equilibria
    stable_x, stable_y = [], []
    unstable_x, unstable_y = [], []
    saddle_x, saddle_y = [], []

    for eq in dynamic_equilibria:
        x, y = eq['point']
        if eq['stability'] == 'stable':
            stable_x.append(x)
            stable_y.append(y)
        elif eq['stability'] == 'unstable':
            unstable_x.append(x)
            unstable_y.append(y)
        elif eq['stability'] == 'saddle':
            saddle_x.append(x)
            saddle_y.append(y)

    stable_eq_plot.set_data(stable_x, stable_y)
    unstable_eq_plot.set_data(unstable_x, unstable_y)
    saddle_eq_plot.set_data(saddle_x, saddle_y)

    return line2, dynamic_V_nullcline, stable_eq_plot, unstable_eq_plot, saddle_eq_plot, dynamic_limit_cycle




# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(a[0]), interval=0.00001, blit=True)

plt.tight_layout()
plt.show()