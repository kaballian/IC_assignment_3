import math
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# --- Constants ---
q = 1.602e-19       # Electron charge [C]
k = 1.380649e-23    # Boltzmann constant [J/K]
t = 300             # Temperature [K]

# --- Given / user-defined parameters ---
# (fill in actual circuit values)
R6 = 47e3            # [Ohm]
R8 = 100e3            # [Ohm]
R10 = 100e3         # [Ohm]
R11 = 100           # [Ohm]
C2 = 220e-9         # [F]
ft = 2.8e6          # [Hz] - GBW of op amp
A_U4 = 1001         # Closed-loop gain
A_NU3 = 1
A_NU4 = 1 + (R10 / R11)
e_nw = 21e-9        # Op amp voltage noise [V/√Hz]
i_ni = 0.6e-15        # Op amp current noise [A/√Hz]
f_ce = 1000           # 1/f corner frequency [Hz]
I_d = 1e-6          # Photodiode current [A]

# --- Derived frequencies ---
f_U3l = (1 / (2 * math.pi * R6 * C2)) * (math.pi / 2)
f_U3h = (ft / A_NU4) * (math.pi / 2)

f_U4l = 3 * (math.pi / 2)
f_U4h = f_U3h

# --- Noise equations ---
# Shot noise from diode
def V_Ndiode(id):
    return R8 * A_NU3 * A_U4 * math.sqrt(2 * q * id * (f_U3h - f_U3l))

# V_Ndiode = (
#     R8 * A_NU3 * A_U4 *
#     math.sqrt(2 * q * I_d * (f_U3h - f_U3l))
# )

# Voltage noise from U3
V_NU3 = (
    e_nw * A_NU3 * A_U4 *
    math.sqrt(f_ce * math.log(f_U3h / f_U3l) + f_U3h - f_U3l)
)

# Current noise from U3
V_NIU3 = (
    A_NU3 * A_U4 * i_ni * R8 *
    math.sqrt(f_U3h - f_U3l)
)

# Thermal noise from R8
V_NR8 = (
    A_NU3 * A_U4 *
    math.sqrt(4 * k * t * R8 * (f_U3h - f_U3l))
)

# Noise from R6 (U4 related)
V_NR6 = (
    A_U4 *
    math.sqrt(4 * k * t * R6 * (f_U3l - f_U4l))
)

# Voltage noise from U4
V_NU4 = (
    e_nw * A_NU4 *
    math.sqrt(f_ce * math.log(f_U4h / f_U4l) + f_U4h - f_U4l)
)

# Current noise from U4 inputs
V_NIU4_plus = (
    A_NU4 * i_ni * R6 *
    math.sqrt(f_U4h - f_U4l)
)

R10R11_parallel = (R10 * R11) / (R10 + R11)
V_NIU4_minus = (
    A_NU4 * i_ni * R10R11_parallel *
    math.sqrt(f_U4h - f_U4l)
)

# Thermal noise from feedback network
V_NR10R11 = (
    A_NU4 *
    math.sqrt(4 * k * t * R10R11_parallel * (f_U4h - f_U4l))
)

# --- Total noise ---
# V_Ntotal = math.sqrt(
#     V_Ndiode**2 + V_NU3**2 + V_NIU3**2 +
#     V_NR8**2 + V_NR6**2 + V_NU4**2 +
#     V_NIU4_plus**2 + V_NIU4_minus**2 +
#     V_NR10R11**2
# )
def V_Ntotal(id) : 
    return math.sqrt(
        V_Ndiode(id)**2 + V_NU3**2 + V_NIU3**2 +
        V_NR8**2 + V_NR6**2 + V_NU4**2 +
        V_NIU4_plus**2 + V_NIU4_minus**2 +
        V_NR10R11**2
    )


id_values = np.linspace(1e-6, 100e-6, 100) #1ua -> 100uA
data = []

for id in id_values:
    data.append({
        "id [A]": id,
        "VNtotal [V]" : V_Ntotal(id)
    })



# MEASURED STUFF
I_diode_uA = np.array([
    -0.0000081, -0.00000825, -0.00000815, -0.00000463,
     0.000158, 0.0003001, 0.0005042, 0.0007771, 0.0011065,
     0.001941, 0.002965, 0.004147, 0.00544, 0.006854,
     0.008331, 0.01153, 0.01495, 0.01854, 0.02228,
     0.02614, 0.03011, 0.03417, 0.0383, 0.0398,
     0.0398, 0.0398
])
I_diode = I_diode_uA * 1e-6
V_noise_mVAC = np.array([
    3.8, 3.8, 3.8, 3.8, 4.2, 4.2, 4.5, 5.2, 5.5,
    6.5, 8.1, 9.2, 10.2, 11, 12.8, 15, 18, 23,
    21, 23, 24, 25, 27, 54, 54, 56
])
V_noise = V_noise_mVAC * 1e-3
mask = I_diode > 0
I_diode_pos = I_diode[mask]
V_noise_pos = V_noise[mask]

# fig, axs = plt.subplots(2, 1, figsize=(7, 8))


plt.figure()
# plt.plot(I_diode_pos, V_noise_pos, 'o-', label='Measured')
plt.plot(I_diode_uA, V_noise_mVAC, 'o-', label='Measured')
plt.xlabel('Diode current $I_d$ [mA]')
plt.ylabel('Output noise $V_{noise}$ [mV AC-RMS]')
plt.title('Measured Noise vs Diode Current')
plt.grid(True, which='both')
plt.legend()
plt.show()

# axs[0].plot(I_diode_pos, V_noise_pos, 'o-', label='Measured Noise')
# axs[0].set_xlabel('Diode current $I_d$ [A]')
# axs[0].set_ylabel('Output noise $V_{noise}$ [V]')
# axs[0].grid(True, which='both')
# axs[0].legend()
# axs[0].set_title('Measured Noise vs Diode Current')







# --- Print results ---
print("f_U3l = {:.3f} Hz".format(f_U3l))
print("f_U3h = {:.3f} Hz".format(f_U3h))
print("f_U4l = {:.3f} Hz".format(f_U4l))
print("f_U4h = {:.3f} Hz".format(f_U4h))
print()
print("V_Ndiode = {:.3e} V @ 1uA".format(V_Ndiode(1e-6)))
print("V_NU3 = {:.3e} V".format(V_NU3))
print("V_NIU3 = {:.3e} V".format(V_NIU3))
print("V_NR8 = {:.3e} V".format(V_NR8))
print("V_NR6 = {:.3e} V".format(V_NR6))
print("V_NU4 = {:.3e} V".format(V_NU4))
print("V_NIU4+ = {:.3e} V".format(V_NIU4_plus))
print("V_NIU4- = {:.3e} V".format(V_NIU4_minus))
print("V_NR10R11 = {:.3e} V".format(V_NR10R11))
print()
print("Total noise (V_Ntotal) = {:.3e} V @ id=1uA".format(V_Ntotal(1e-6)))

print("data:\n")
df = pd.DataFrame(data)
print(df)


plt.figure()
plt.plot(df["id [A]"], df["VNtotal [V]"], label=r"$V_{Ntotal}$")
plt.xlabel("Diode current $I_d$ [A]")
plt.ylabel("Total noise $V_{Ntotal}$ [V]")
plt.title("Total Noise vs Diode Current")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.show()


# axs[1].plot(df["id [A]"], df["VNtotal [V]"], label=r"$V_{Ntotal}$")
# axs[1].set_xlabel("Diode current $I_d$ [A]")
# axs[1].set_ylabel("Total noise $V_{Ntotal}$ [V]")
# # axs[1].title("Total Noise vs Diode Current")
# axs[1].grid(True, which="both", ls="--")
# axs[1].legend()
# plt.tight_layout()
# plt.show()