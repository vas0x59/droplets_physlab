import numpy as np
import scipy.integrate
import cantera as ct

import matplotlib.pyplot as plt
from scipy.stats import lognorm


MODE =1 # 0 -- inf 1 -- one_third_rule(x) 2 -- one_third_rule(T)
RM = False

"""
x - [T, m]

"""




MILLI = 10**-3
T_C0 = 273.15
RGC = 8.31
P = 10**5


# water
rho_l = 1000 
c_l = 4200 
h_gl = 40.66 * 1000
M = 18/1000
SP = "H2O"
mech = "gri30.yaml"
SP_O2 = "O2"
SP_N2 = "N2"

def P_sat(T):
    return  10**5 * np.exp(h_gl/8.31 * (1/(273.15+100) - 1/T))
print(P_sat(273+20))
# Ethanol
# rho_l = 1000 
# M = 46/1000
# c_l = 	112.4 / M
# h_gl = 42.3 * 1000

# SP = "c2h5oh"
# mech = "/Users/vasily/Downloads/ethanol-marinov.yml"
# SP_O2 = "o2"
# SP_N2 = "n2"





# water = ct.Water()
# water.P_sat()
v =0# m/s
T_inf = 31 + T_C0
T_0 =20 + T_C0


L = 0.03

Y_inf = P_sat(T_inf)*18*0.42/ (29*10**5)
print("Y_inf", Y_inf)
ct_inf_sol = ct.Solution(mech)
ct_s_d = ct.Solution(mech)
ct_m_d = ct.Solution(mech)


Y_air_N2 = 0.8
Y_air_O2 = 0.2


def calc_Y(Y_SP):
    return {SP_N2 : Y_air_N2*(1-Y_SP), SP_O2: Y_air_O2*(1-Y_SP), SP: Y_SP}


ct_inf_sol.TPY = T_inf, P, calc_Y(Y_inf)
SP_ind = ct_inf_sol.species_index(SP)

C_inf = ct_inf_sol.concentrations[SP_ind]*1000
mu_inf = ct_inf_sol.viscosity
rho_inf = ct_inf_sol.density_mass
cp_inf = ct_inf_sol.cp_mass
k_inf = ct_inf_sol.thermal_conductivity
D_inf = ct_inf_sol.mix_diff_coeffs[SP_ind]


def one_third_rule(x_s, x_inf):
    return x_s + 1/3 * (x_inf - x_s)

def x_dot( t, x):
    global RM
    T, m = x
    print(T, m)
    d = np.cbrt(6*m/rho_l/np.pi)
    if d < 1e-5 or T < 200:
        return np.zeros((2))


    Tc = T - 273.15
    
    # P_sat =  1000 * 0.61094 * np.exp((17.625*Tc) / (Tc + 243.04))
    # P_sat =  0.0582625 	*10**5 * np.exp(h_gl/8.31 * (1/(273.15+20) - 1/T))
    C_s = P_sat(T) / (T * RGC)

    
    # from cantera

    # print(P_sat, P)

    
    С_all = P / (T * RGC)
    
    Y_s = C_s*M /  (  С_all *29e-3 )

    # print("Y_s", Y_s)

    

    
    # if MODE == 2:
    #     ct_m_d.TPX = one_third_rule(T, T_inf), P,  {SP_N2: one_third_rule(0.8*(C_all - C_s), 0.78), SP_O2: one_third_rule(0.2*(C_all - C_s), 0.2), "SP": one_third_rule(C_s, 0.02)}

    #     C_m_ = ct_m_d.concentrations[SP_ind]*1000
    #     # print(C_s_, C_s)
    #     mu_m = ct_m_d.viscosity
    #     rho_m = ct_m_d.density_mass
    #     cp_m = ct_m_d.cp_mass
    #     k_m = ct_m_d.thermal_conductivity
    #     D_m = ct_m_d.mix_diff_coeffs[SP_ind]

    # if MODE == 1:
    ct_s_d.TPY =T, P, calc_Y(Y_s)
    C_s_ = ct_s_d.concentrations[SP_ind]*1000
    print(C_s_, C_s)
    mu_s = ct_s_d.viscosity
    rho_s = ct_s_d.density_mass
    cp_s = ct_s_d.cp_mass
    k_s = ct_s_d.thermal_conductivity
    D_s = ct_s_d.mix_diff_coeffs[SP_ind]
    mu_m = one_third_rule(mu_s, mu_inf) 
    rho_m = one_third_rule(rho_s, rho_inf) 
    cp_m = one_third_rule(cp_s, cp_inf) 
    k_m = one_third_rule(k_s, k_inf) 
    D_m = one_third_rule(D_s,D_inf) 

    # if MODE == 0:
    # mu_m =mu_inf
    # rho_m =rho_inf
    # cp_m = cp_inf
    # k_m = k_inf
    # D_m = D_inf


    
    

    Re = rho_m*d*v/mu_m
    print("Re", Re)
    Pr = cp_m*mu_m/k_m
    Sc = mu_m/(rho_m * D_m)
    B_m = (Y_inf - Y_s)/(Y_s - 1)
    if RM:
        # Ranz Msarshal ?
        Nu = 2 + 0.6*np.sqrt(Re)*np.cbrt(Pr)
        Sh = 2 + 0.6*np.sqrt(Re)*np.cbrt(Sc)


    # q
        print("np.log(1+B_m)", np.log(1+B_m), (Y_inf - Y_s))
        q = (Nu *k_m) * np.pi * d * (T_inf - T)
        g = (Sh*D_m) * np.pi * d * rho_m * -(Y_inf - Y_s)

    else:
       
        # print("B_m", B_m)
        print("np.log(1+B_m)", np.log(1+B_m), (Y_inf - Y_s))
        g = 2*np.pi * d * D_m * rho_m * -(Y_inf - Y_s)
        # B_T = B_m
        q = 2*np.pi * d * k_m * (T_inf - T) 

    # print(C_s, C_inf)
    
    dm = -g
    # dT = q/m/c_l + (h_gl/M/c_l)*dm/m
    dT = 0
    

    
    return np.array([
        dT,
        dm
    ])

def solve_for_d(D_0, t):
    
    m_0 = rho_l * D_0**3 * np.pi * 1/6

    x0 = np.array([T_0, m_0])
    # print(x0)
    # ts = np.linspace(0, t, 600)
    sol = scipy.integrate.solve_ivp(x_dot, (0, t), x0, atol=np.array([1e-12, 1e-12]), rtol=1e-12)
    ts = sol.t
    xs = sol.y.T
    m = xs[:, 1]
    T = xs[:, 0]
    d = np.cbrt(6*m/rho_l/np.pi)
    # print(d[0], d[-1])
    return ts, T, d
D0 =np.sqrt(0.3e-6)*2
v = 0.15
# D0 = 0.09e-3/2
MODE = 1
RM = False
ts_m0, T_m0, d_m0 = solve_for_d(D0, 12*60)
# np.save("t_no_flow", ts_m0)
# np.save("r_no_flow", d_m0/2)
MODE = 1
RM = True
ts_m1, T_m1, d_m1 = solve_for_d(D0, 12*60)
np.save("t_with_flow", ts_m1)
np.save("r_with_flow", d_m1/2)
# np.save("t_no_flow", ts_m0)
# np.save("r_no_flow", d_m0/2)
# MODE = 2
# ts_m2, T_m2, d_m2 = solve_for_d(D0)

# # print(sol)
plt.figure()
plt.plot(ts_m0, T_m0, label="Temp, K 0")
plt.plot(ts_m1, T_m1, label="Temp, K 1 RM")
# plt.plot(ts_m2, T_m2, label="Temp, K 2")
plt.legend()

plt.figure()
plt.plot(ts_m0, (d_m0*1000/2), label="mm 0")
plt.plot(ts_m1, (d_m1*1000/2), label=" mm 1 RM")
# plt.plot(ts_m2, d_m2*1000, label="diameter, mm 2")
plt.legend()
plt.figure()

print(np.polyfit(ts_m0, np.power(d_m0*1000/2, 2), 1))
plt.plot(ts_m0, np.power(d_m0*1000/2, 2), label="mm^2 0")
plt.plot(ts_m1, np.power(d_m1*1000/2, 2), label=" mm^2 1 RM")
# plt.plot(ts_m2, d_m2*1000, label="diameter, mm 2")
plt.legend()
plt.figure()
plt.plot(ts_m0, np.power(d_m0*1000/2, 3/2), label="mm^3/2 0")
plt.plot(ts_m1, np.power(d_m1*1000/2, 3/2), label=" mm^3/2 1 RM")
# plt.plot(ts_m2, d_m2*1000, label="diameter, mm 2")
plt.legend()
plt.show()
exit(0)

RM = True
MODE = 1
# distr = lognorm(s=0.8, scale=0.2)
# ds = np.linspace(0, 1)
# plt.figure()
# plt.plot(ds, distr.pdf(ds))

# print()

T_19 = np.load("/Users/vasily/Downloads/Telegram Desktop/68/18/Rs.npy") / 160 /1000
T_16 = np.load("/Users/vasily/Downloads/Telegram Desktop/68/16/Rs.npy") / 160 /1000

th = 0.05*1e-3
print(th, th)




# print(np.quantile(T_16, 0.1))
# print(np.min(T_16), np.min(T_19))


density = True

d_T_out = np.array([(lambda res: [res[2][-2], res[1][-2]])(solve_for_d(d, L/v)) for d in T_19])
d_T_out = d_T_out[d_T_out[:, 0] > th]

T_19 = T_19[T_19 > th]
T_16 = T_16[T_16 > th]
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



