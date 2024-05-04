import numpy as np
import scipy.integrate
import cantera as ct

import matplotlib.pyplot as plt
from scipy.stats import lognorm


MODE = 0 # 0 -- inf 1 -- one_third_rule(x) 2 -- one_third_rule(T)


"""
x - [T, m]

"""

MILLI = 10**-3
T_C0 = 273.15
RGC = 8.31

rho_l = 1000 # water
c_l = 4200 # water

# water = ct.Water()
# water.P_sat()
v = 0.2 # m/s
T_inf = 30 + T_C0
T_0 = 30 + T_C0
M = 18/1000
h_gl = 40.66 * 1000
L = 0.02

ct_inf_sol = ct.Solution('gri30.yaml')
ct_s_d = ct.Solution('gri30.yaml')
ct_m_d = ct.Solution('gri30.yaml')
ct_inf_sol.TPX = T_inf, 10**5, {"N2" : 0.78, "O2": 0.2, "H2O": 0.02}
H2O_ind = ct_inf_sol.species_index("H2O")

C_inf = ct_inf_sol.concentrations[H2O_ind]*1000
mu_inf = ct_inf_sol.viscosity
rho_inf = ct_inf_sol.density_mass
cp_inf = ct_inf_sol.cp_mass
k_inf = ct_inf_sol.thermal_conductivity
D_inf = ct_inf_sol.mix_diff_coeffs[H2O_ind]


def one_third_rule(x_s, x_inf):
    return x_s + 1/3 * (x_inf - x_s)

def x_dot(x, t):
    T, m = x
    Tc = T - 273.15
    
    P_sat =  1000 * 0.61094 * np.exp((17.625*Tc) / (Tc + 243.04))
    C_s = P_sat / (T * RGC)
    
    # from cantera



    
    C_all = 10**5 / (T * RGC)
    

    
    if MODE == 2:
        ct_m_d.TPX = one_third_rule(T, T_inf), 10**5,  {"N2": one_third_rule(0.8*(C_all - C_s), 0.78), "O2": one_third_rule(0.2*(C_all - C_s), 0.2), "H2O": one_third_rule(C_s, 0.02)}

        C_m_ = ct_m_d.concentrations[H2O_ind]*1000
        # print(C_s_, C_s)
        mu_m = ct_m_d.viscosity
        rho_m = ct_m_d.density_mass
        cp_m = ct_m_d.cp_mass
        k_m = ct_m_d.thermal_conductivity
        D_m = ct_m_d.mix_diff_coeffs[H2O_ind]

    if MODE == 1:
        ct_s_d.TPX =T, 10**5,  {"N2": 0.8*(C_all - C_s), "O2": 0.2*(C_all - C_s), "H2O": C_s}
        C_s_ = ct_s_d.concentrations[H2O_ind]*1000
        # print(C_s_, C_s)
        mu_s = ct_s_d.viscosity
        rho_s = ct_s_d.density_mass
        cp_s = ct_s_d.cp_mass
        k_s = ct_s_d.thermal_conductivity
        D_s = ct_s_d.mix_diff_coeffs[H2O_ind]
        mu_m = one_third_rule(mu_s, mu_inf) 
        rho_m = one_third_rule(rho_s, rho_inf) 
        cp_m = one_third_rule(cp_s, cp_inf) 
        k_m = one_third_rule(k_s, k_inf) 
        D_m = one_third_rule(D_s,D_inf) 

    if MODE == 0:
        mu_m =mu_inf
        rho_m =rho_inf
        cp_m = cp_inf
        k_m = k_inf
        D_m = D_inf


    
    
    
    d = np.cbrt(6*m/rho_l/np.pi)

    Re = rho_m*d*v/mu_m
    Pr = cp_m*mu_m/k_m
    Sc = mu_m/(rho_m * D_m)

    # Ranz Marshal ?
    Nu = 2 + 0.6*np.sqrt(Re)*np.cbrt(Pr)
    Sh = 2 + 0.6*np.sqrt(Re)*np.cbrt(Sc)




    q = (Nu *k_m) * np.pi * d * (T - T_inf)
    g = M*(Sh*D_m) * np.pi * d * (C_s - C_inf)

    # print(C_s, C_inf)
    
    dm = -g
    dT = -q/m/c_l + (h_gl/c_l)*dm/m

    
    return np.array([
        dT,
        dm
    ])

def solve_for_d(D_0):
    
    m_0 = rho_l * D_0**3 * np.pi * 1/6

    x0 = np.array([T_0, m_0])
    # print(x0)
    ts = np.linspace(0, L/v, 100)
    sol = scipy.integrate.odeint(x_dot, x0, ts)
    m = sol[:, 1]
    T = sol[:, 0]
    d = np.cbrt(6*m/rho_l/np.pi)
    # print(d[0], d[-1])
    return ts, T, d
D0 = 0.1*10**-3
MODE = 0
ts_m0, T_m0, d_m0 = solve_for_d(D0)
MODE = 1
ts_m1, T_m1, d_m1 = solve_for_d(D0)
MODE = 2
ts_m2, T_m2, d_m2 = solve_for_d(D0)

# # print(sol)
plt.figure()
plt.plot(ts_m0, T_m0, label="Temp, K 0")
plt.plot(ts_m1, T_m1, label="Temp, K 1")
plt.plot(ts_m2, T_m2, label="Temp, K 2")
plt.legend()

plt.figure()
plt.plot(ts_m0, d_m0*1000, label="diameter, mm 0")
plt.plot(ts_m1, d_m1*1000, label="diameter, mm 1")
plt.plot(ts_m2, d_m2*1000, label="diameter, mm 2")
plt.legend()
# plt.show()
# exit(0)


# distr = lognorm(s=0.8, scale=0.2)
# ds = np.linspace(0, 1)
# plt.figure()
# plt.plot(ds, distr.pdf(ds))

# print()

T_19 = np.load("/Users/vasily/Downloads/Telegram Desktop/68/18/Rs.npy") / 150 /1000
T_16 = np.load("/Users/vasily/Downloads/Telegram Desktop/68/16/Rs.npy") / 150 /1000
# print(np.min(T_16), np.min(T_19))


density = True

d_T_out = np.array([(lambda res: [res[2][-2], res[1][-2]])(solve_for_d(d)) for d in T_19])
d_T_out = d_T_out[d_T_out[:, 0] > np.min(T_16)]
plt.figure()
plt.hist(T_19*1000, bins=np.linspace(0,1, 30), density=density, color="g", alpha=0.5, label="before", histtype=u'step', linewidth=2)
plt.hist(d_T_out[:, 0]*1000, bins=np.linspace(0,1, 30), density=density, color="r", alpha=0.5, label="after sim", histtype=u'step', linewidth=2)
plt.hist(T_16*1000, bins=np.linspace(0,1, 30), density=density, color="b", alpha=0.5, label="after experiment", histtype=u'step', linewidth=2)
plt.legend()

plt.figure()
plt.hist(d_T_out[:, 1], bins=20, density=density, color="r", alpha=0.5, label="after", histtype=u'step', linewidth=2)

plt.show()
# # print(sol)
# plt.figure()
# plt.plot(ts, T, label="Temp, K")
# plt.legend()

# plt.figure()
# plt.plot(ts, d*1000, label="diameter, mm")
# plt.legend()






# plt.show()
# 



