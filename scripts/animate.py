import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from neuron_model import NeuronModel

# Simulating neuron dynamics
sub_hop = NeuronModel('subcritical_Hopf')
dt = 0.01   
T = 100
t = np.arange(0, T, dt)
I_step = sub_hop.create_step_current(t, 1, 80, 0, 5)
I_ramp = sub_hop.create_ramp_current(t, 1, 40, 0, 10)
pulse_times = [0, 6] #[0, 10, 30, 40, 50, 60, 70, 80, 90]  
I_pulse = sub_hop.create_pulse_train(t, pulse_times, 3, 0, 50)
I_ext = I_pulse
a = sub_hop.simulate(T, dt, [-70, 0], I_ext)    

fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle("subcritical_Hopf Bifurcation model")
# axs[0].plot(a[0], a[1][:, 0])
axs[0].set_xticks([])
axs[0].set_ylabel("Membrane potential (mV)")
axs[0].set_ylim([-80, 20])  
# axs[1].plot(a[0], I_ext(np.arange(0, T, dt)))   
axs[1].set_xlabel("Time (ms)")
axs[1].set_ylabel("I_ext ($uA/cm^2$)")

# Line to update
line0, = axs[0].plot(a[0], a[1][:, 0], marker = 'o', markevery = [-1])
line1, = axs[1].plot(a[0], I_ext(np.arange(0, T, dt)), marker = 'o', markevery = [-1])


def update(frame):
    line0.set_data(a[0][:frame],  a[1][:, 0][:frame])
    line1.set_data(a[0][:frame], I_ext(np.arange(0, T, dt)[:frame]) )

    return line0, line1

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(a[0]), interval=0.001, blit=True)

plt.show()

# # Extract phase portrait data
# n_values = a[1][:, 1]

# fig, ax = plt.subplots()
# ax.set_xlim(min(V_values) - 5, max(V_values) + 5)
# ax.set_ylim(min(n_values) - 0.1, max(n_values) + 0.1)
# ax.set_xlabel("Membrane potential (mV)")
# ax.set_ylabel("n (Gating variable)")
# ax.set_title("Phase Portrait Animation")

# # Line to update
# phase_line, = ax.plot([], [], lw=2)

# def update_phase(frame):
#     phase_line.set_data(V_values[:frame], n_values[:frame])
#     return phase_line,

# # Create animation
# phase_ani = animation.FuncAnimation(fig, update_phase, frames=len(V_values), interval=1, blit=True)
# plt.show()
