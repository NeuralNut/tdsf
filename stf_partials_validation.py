import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch

# Define the dictionary with constants for each material
constants_dict = {
    "Tuff/Rhyolite": {
        "h0": 122.0, "R0": 202.0, "gamma0": 26.0, "P10": 3.6E6, "P20": 5.0E6, "pv": 3500.0, "sv": 2021.0, "n": 2.4, "rho": 2000.0
    },
    "Granite": {
        "h0": 122.0, "R0": 321.0, "gamma0": 34.0, "P10": 4.6E6, "P20": 2.4E6, "pv": 5500.0, "sv": 3175.0, "n": 2.4, "rho": 2550.0
    },
    "Salt": {
        "h0": 122.0, "R0": 478.0, "gamma0": 31.0, "P10": 5.5E6, "P20": 0.8E6, "pv": 4670.0, "sv": 2696.0, "n": 1.87, "rho": 2200.0
    },
    "Shale": {
        "h0": 122.0, "R0": 265.0, "gamma0": 42.0, "P10": 4.2E6, "P20": 2.5E6, "pv": 4320.0, "sv": 2495.0, "n": 2.4, "rho": 2350.0
    },
    "Wet Tuff": {
        "h0": 122.0, "R0": 202.0, "gamma0": 26.0, "P10": 3.6E6, "P20": 5.0E6, "pv": 2400.0, "sv": 1300.0, "n": 2.4, "rho": 1915.0
    },
    "Wet Granite": {
        "h0": 122.0, "R0": 321.0, "gamma0": 34.0, "P10": 4.6E6, "P20": 2.4E6, "pv": 5350.0, "sv": 2795.0, "n": 2.4, "rho": 2650.0
    }
}

# Function to populate variables from the dictionary based on material name
def get_constants(material_name, device):
    if material_name in constants_dict:
        material_constants = constants_dict[material_name]
        h0 = torch.tensor(material_constants["h0"], requires_grad=False, device=device)
        R0 = torch.tensor(material_constants["R0"], requires_grad=False, device=device)
        gamma0 = torch.tensor(material_constants["gamma0"], requires_grad=False, device=device)
        P10 = torch.tensor(material_constants["P10"], requires_grad=False, device=device)
        P20 = torch.tensor(material_constants["P20"], requires_grad=False, device=device)
        pv = torch.tensor(material_constants["pv"], requires_grad=False, device=device)
        sv = torch.tensor(material_constants["sv"], requires_grad=False, device=device)
        n = torch.tensor(material_constants["n"], requires_grad=False, device=device)
        rho = torch.tensor(material_constants["rho"], requires_grad=False, device=device)
        return h0, R0, gamma0, P10, P20, pv, sv, n, rho
    else:
        raise ValueError(f"Material '{material_name}' not found in the constants dictionary.")



def compute_partial_derivatives(w=1.0, h=150.0, material='Tuff/Rhyolite', device='cpu', plot=False):
    h0, R0, gamma0, P10, P20, pv, sv, n, rho = get_constants(material, device)
    
    mu = rho*(sv**2)
    lam = rho*(pv**2)-2*mu   

    # Define h, w, t as tensors with requires_grad=True to compute derivatives
    h = torch.tensor(h, requires_grad=True, device=device) 
    w = torch.tensor(w, requires_grad=True, device=device)  
    t = torch.linspace(0, 4.0, steps=4000, requires_grad=True, device=device) 
    def Heaviside(t):
        return torch.where(t > 0, torch.tensor(1.0), torch.tensor(0.0))

    # Define H(t), Heaviside step function
    H_t = Heaviside(t)

    # Compute R_el based on equation (7) and (8)
    R_el = R0 * (h0 / h).pow(1/n) * w.pow(1/3)

    # Compute gamma based on equation in the screenshot
    gamma = gamma0 * (R0 / R_el)

    # Compute beta based on the given equation
    beta = (lam + 2 * mu) / (4 * mu)

    # Compute w0 based on the given equation
    w0 = pv / R_el

    # Compute alpha and p based on the given equations
    alpha = w0 / (2 * beta)

    p = w0 * torch.sqrt(1 / (2 * beta) - 1 / (4 * beta**2))

    # Debugging: Print intermediate values
    print(f'R_el: {R_el}')
    print(f'gamma: {gamma}')
    print(f'beta: {beta}')
    print(f'w0: {w0}')
    print(f'alpha: {alpha}')
    print(f'p: {p}')
    print(f'(R_el*pv): {(R_el*pv):.2f}')

    # Define f(t) based on equation (3)
    def f_t(R_el, t):
        return (R_el / (4 * alpha)) * (pv**2 / (beta * p)) * H_t * torch.exp(-alpha * t) * torch.sin(p * t)
    
    def df_t_dt(R_el, mu, pv, beta, p, alpha, t):
        A = (R_el / (4 * mu)) * (pv**2 / (beta * p))
        return A * H_t * torch.exp(-alpha * t) * (-alpha * torch.sin(p * t) + p * torch.cos(p * t))

    # Define B(t) based on equation (2)
    def B_t(R_el, t):
        P0_t = P10 * (h / h0)  # P1 - P2 with P1 and P2
        P2_t = P20 * (h0 / h).pow(1/3) * (R0 / R_el).pow(3) * w.pow(0.87)
        return (torch.exp(-gamma * t) * P0_t + P2_t) * H_t


    def dB_t_dRel_old():
        # Compute the terms of the expression
        term1 = (gamma * torch.exp(-gamma * t) / R_el) * P10 * (h / h0)
        term2 = (torch.exp(-gamma * t) - 3 * (1 - torch.exp(-gamma * t))) / R_el
        term3 = P20 * (h0 / h).pow(1/3) * (R0 / R_el).pow(3) * w.pow(0.87)
        
        # Combine the terms
        result = (term1 - term2 * term3) * H_t
        
        return result


    def dB_t_dRel_new():
        term1 = -torch.exp(-gamma * t) * P10 * (h / h0) * (t * gamma / R_el - n / R_el * (h / h0).pow(1/n)) * (h * w.pow(-1/3)).pow(1/3)
        term2 = P20 * w.pow(0.87) * ((t * torch.exp(-gamma * t) * R0.pow(3)) / R_el.pow(4) * (h0 / h).pow(1/3) - (t * torch.exp(-gamma * t) * R0.pow(3)) / R_el.pow(4) * (h0 / h).pow(1/3))
        term3 = (1 - torch.exp(-gamma * t)) * (1 / 3 * n * R0.pow(2) / R_el.pow(3) * (h0 / h).pow(1/3) * w.pow(-2/3) * (h * w.pow(-1/3)).pow(1/3))
        term4 = (1 - torch.exp(-gamma * t)) * (-3 * R0.pow(3) / R_el.pow(4) * (h0 / h).pow(1/3) * (h * w.pow(-1/3)).pow(1/3))
        return (term1 + term2 + term3 + term4) * H_t


    def dRel_dw(R0, h0, h, n, w):
        return R0 * (h0 / h).pow(1/n) * (1 / 3) * w.pow(-2/3)

    def dRel_dh(R0, h0, h, n, w):
        return -R0 * (1 / n) * (h0 / h).pow(1/n) * (1 / h) * w.pow(1/3)

    # Compute the time derivative of f(t)
    f_t_val = f_t(R_el, t)
    df_dt = df_t_dt(R_el, mu, pv, beta, p, alpha, t)
    

    # Reshape for conv1d: [batch_size, channels, sequence_length]
    df_dt_reshaped = df_dt.view(1, 1, -1)

    B_t_val = B_t(R_el, t)
    B_t_reshaped = B_t_val.view(1, 1, -1)


    S_t = F.conv1d(-df_dt_reshaped/(R_el*pv), B_t_reshaped, padding=B_t_reshaped.size(-1)-1) # full convolution padding
    S_t = S_t.view(-1)[t.size(0)-1:]  # Ensure S_t has the same length as t

    # S_t = F.conv1d(df_dt_reshaped, B_t_reshaped, padding=B_t_reshaped.size(-1)//2) # half convolution padding
    # S_t = S_t.view(-1)[:t.size(0)]  # Ensure S_t has the same length as t


    # Compute dS(t)/dW and dS(t)/dh using autograd
    dS_dW_num = torch.autograd.grad(S_t.sum(), w, create_graph=True)[0]
    dS_dh_num = torch.autograd.grad(S_t.sum(), h, create_graph=True)[0]
    df_dt_num = torch.autograd.grad(f_t_val.sum(), t, create_graph=True)[0]
    
    # dB_t_dRel_val = dB_t_dRel_old()
    dB_t_dRel_val = dB_t_dRel_new()
    dB_t_dRel_reshaped = dB_t_dRel_val.view(1, 1, -1)
    
    # Perform convolution using conv1d
    dS_t_dRel = F.conv1d(-df_dt_reshaped/(R_el*pv), dB_t_dRel_reshaped, padding=dB_t_dRel_reshaped.size(-1)-1) # full convolution padding
    dS_t_dRel = dS_t_dRel.view(-1)[t.size(0)-1:]  # Ensure S_t has the same length as t

    # dS_t_dRel = F.conv1d(df_dt_reshaped, dB_t_dRel_reshaped, padding=dB_t_dRel_reshaped.size(-1)//2) # half convolution padding
    # dS_t_dRel = dS_t_dRel.view(-1)[:t.size(0)]  # Ensure S_t has the same length as t
    

    dS_dW_ana = dS_t_dRel * dRel_dw(R0, h0, h, n, w)
    dS_dh_ana = dS_t_dRel * dRel_dh(R0, h0, h, n, w)

    t_cpu = t.detach().cpu().numpy()
    
    S_t_cpu = S_t.detach().cpu().numpy()
    B_t_cpu = B_t_val.detach().cpu().numpy()
    f_t_cpu = f_t_val.detach().cpu().numpy()
    
    dB_t_dRel_cpu = dB_t_dRel_val.detach().cpu().numpy()
    dS_t_dRel_cpu = dS_t_dRel.detach().cpu().numpy()
    df_dt_cpu = df_dt.detach().cpu().numpy()
    dS_dW_ana_cpu = dS_dW_ana.detach().cpu().numpy()
    dS_dh_ana_cpu = dS_dh_ana.detach().cpu().numpy()
    df_dt_num_cpu = df_dt_num.detach().cpu().numpy()

    
    if plot:
        # Convert S_t, f_t_val, and B_t_val to CPU for plotting
        
        # Plot S(t) on primary y-axis and f(t) on secondary y-axis
        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.grid(True)
        ax1.plot(t_cpu, S_t_cpu, label='S(t)', color='tab:blue')
        ax1.set_xlabel('Time (t)')
        ax1.set_ylabel('S(t)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(t_cpu, dS_t_dRel_cpu, label='dS(t)/dRel', color='tab:orange')
        ax2.set_ylabel('dS(t)/dRel', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.title(f'Plots of S(t) and dS(t)/dRel at w={w.item()}, h={h.item()}')
        plt.show()

        # Plot B(t) on primary y-axis and f(t) on secondary y-axis
        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.grid(True)
        ax1.plot(t_cpu, B_t_cpu, label='B(t)', color='tab:blue')
        ax1.set_xlabel('Time (t)')
        ax1.set_ylabel('B(t)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(t_cpu, dB_t_dRel_cpu, label='dB(t)/dRel', color='tab:orange')
        ax2.set_ylabel('dB(t)/dRel', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.title(f'Plots of B(t) and dB(t)/dRel at w={w.item()}, h={h.item()}')
        plt.show()

        # Plot f(t) on primary y-axis and df(t)/dt on secondary y-axis
        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.grid(True)
        ax1.plot(t_cpu, f_t_cpu, label='f(t)', color='tab:blue')
        ax1.set_xlabel('Time (t)')
        ax1.set_ylabel('f(t)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(t_cpu, df_dt_cpu, label='df(t)/dt', color='tab:orange')
        ax2.set_ylabel('df(t)/dt', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.title(f'Plots of f(t) and df(t)/dt at w={w.item()}, h={h.item()}')
        plt.show()


        # Plot f(t) on primary y-axis and df(t)/dt on secondary y-axis
        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.grid(True)
        ax1.plot(t_cpu, dS_dW_ana_cpu, label='dS/dW', color='tab:blue')
        ax1.set_xlabel('Time (t)')
        ax1.set_ylabel('dS/dW', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(t_cpu, dS_dh_ana_cpu, label='dS/dh', color='tab:orange')
        ax2.set_ylabel('dS/dh', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.title(f'Plots of dS/dW and dS/dh at W={w.item()}, h={h.item()}')
        plt.show()


        print('Analytical')
        print(f'dS/dW: {dS_dW_ana.detach().cpu().numpy().sum()}')
        print(f'dS/dh: {dS_dh_ana.detach().cpu().numpy().sum()}')
        print('Numerical')
        print(f'dS/dW: {dS_dW_num}')
        print(f'dS/dh: {dS_dh_num}')

        out = {
                'dS_dW_ana': dS_dW_ana_cpu,
                'dS_dh_ana': dS_dh_ana_cpu,
                'df_dt': df_dt_cpu,
                'S_t': S_t_cpu,
                'f_t': f_t_cpu,
                'B_t': B_t_cpu,
                't': t_cpu,
                'dB_t_dRel': dB_t_dRel_cpu,
                'dS_t_dRel': dS_t_dRel_cpu,
                'R_el*pv': (R_el*pv).item(),
                'h': h.item(),
                'w': w.item()
                }

        return out
    else:
        out = {
                'dS_dW_ana': dS_dW_ana_cpu,
                'dS_dh_ana': dS_dh_ana_cpu,
                'df_dt': df_dt_cpu,
                'S_t': S_t_cpu,
                'f_t': f_t_cpu,
                'B_t': B_t_cpu,
                't': t_cpu,
                'dB_t_dRel': dB_t_dRel_cpu,
                'dS_t_dRel': dS_t_dRel_cpu,
                'R_el*pv': (R_el*pv).item(),
                'h': h.item(),
                'w': w.item()
                }
        return out


plot = True
material = 'Tuff/Rhyolite' # Tuff/Rhyolite, Granite, Salt, Shale, Wet Tuff, Wet Granite 
# Example usage on GPU (if available)
if torch.cuda.is_available():
    result = compute_partial_derivatives(w=1.0, h=150.0, device='cuda', material=material, plot=plot)
else:
    result = compute_partial_derivatives(w=1.0, h=150.0, device='cpu' , material=material, plot=plot)


print(result)


# compare to scipy convolve function
from scipy.signal import convolve
dfdt = result['df_dt']
Bt = result['B_t']

St = convolve(dfdt, Bt, mode='full')
plt.plot(St)
plt.show()


import itertools
ws = [1.0,  2.0]
hs = [100.0, 900.0]
wh_combinations = list(itertools.product(ws, hs))
out = [[],[],[],[]]
for w,h in wh_combinations:
    result = compute_partial_derivatives(w=w, h=h, device='cuda', plot=False)
    out[0].append(result['dS_dW_ana'])
    out[1].append(result['dS_dh_ana'])
    out[2].append(result['w'])
    out[3].append(result['h'])
    


fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=False)
for i in range(len(ws)*len(hs)):
    axs[0].plot(out[0][i], label=f'w: {out[2][i]}, h: {out[3][i]}')
    axs[1].plot(out[1][i], label=f'w: {out[2][i]}, h: {out[3][i]}')
axs[0].set_title('dS/dW')
axs[1].set_title('dS/dh')
axs[0].legend()
axs[1].legend()

plt.tight_layout()
# plt.show()
plt.savefig('dS_dW_dS_dh.png')



ws = [1.0,  2.0]
hs = [100.0, 900.0]
wh_combinations = list(itertools.product(ws, hs))
out = [[],[],[],[]]
for w,h in wh_combinations:
    result = compute_partial_derivatives(w=w, h=h, device='cuda', plot=False)
    out[0].append(result['f_t'])
    out[1].append(result['df_dt'])
    out[2].append(result['w'])
    out[3].append(result['h'])
    

# make a 2 column, 1 row subplot figure, where columns are dS/dW and dS/dh, and (W,h) combinations are different line colors
fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=False)
for i in range(len(ws)*len(hs)):
    axs[0].plot(out[0][i], label=f'w: {out[2][i]}, h: {out[3][i]}')
    axs[1].plot(out[1][i], label=f'w: {out[2][i]}, h: {out[3][i]}')
axs[0].set_title('f(t)')
axs[1].set_title('df/dt')
axs[0].legend()
axs[1].legend()

plt.tight_layout()
# plt.show()
plt.savefig('f_t_df_dt.png')


ws = [1.0,  2.0]
hs = [100.0, 900.0]
wh_combinations = list(itertools.product(ws, hs))
out = [[],[],[],[]]
for w,h in wh_combinations:
    result = compute_partial_derivatives(w=w, h=h, device='cuda', plot=False)
    out[0].append(result['B_t'])
    out[1].append(result['dB_t_dRel'])
    out[2].append(result['w'])
    out[3].append(result['h'])
    

# make a 2 column, 1 row subplot figure, where columns are dS/dW and dS/dh, and (W,h) combinations are different line colors
fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=False)
for i in range(len(ws)*len(hs)):
    axs[0].plot(out[0][i], label=f'w: {out[2][i]}, h: {out[3][i]}')
    axs[1].plot(out[1][i], label=f'w: {out[2][i]}, h: {out[3][i]}')
axs[0].set_title('B(t)')
axs[1].set_title('dB(t)/dRel')
axs[0].legend()
axs[1].legend()

plt.tight_layout()
# plt.show()

plt.savefig('B_t_dB_t_dRel.png')

ws = [1.0,  2.0]
hs = [100.0, 900.0]
wh_combinations = list(itertools.product(ws, hs))
out = [[],[],[],[]]
for w,h in wh_combinations:
    result = compute_partial_derivatives(w=w, h=h, device='cuda', plot=False)
    out[0].append(result['S_t'])
    out[1].append(result['dS_t_dRel'])
    out[2].append(result['w'])
    out[3].append(result['h'])
    

# make a 2 column, 1 row subplot figure, where columns are dS/dW and dS/dh, and (W,h) combinations are different line colors
fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=False)
for i in range(len(ws)*len(hs)):
    axs[0].plot(out[0][i], label=f'w: {out[2][i]}, h: {out[3][i]}')
    axs[1].plot(out[1][i], label=f'w: {out[2][i]}, h: {out[3][i]}')
axs[0].set_title('S(t)')
axs[1].set_title('dS(t)/dRel')
axs[0].legend()
axs[1].legend()

plt.tight_layout()
# plt.show()

plt.savefig('S_t_dS_t_dRel.png')



import pandas as pd
import os

root = os.path.abspath('')
path = f'{root}/data'

filename = 'tdsf_002_train.pickle'
df_train_all = pd.read_pickle(path + '/' + filename)

filename = 'tdsf_002_val.pickle'
df_test_all = pd.read_pickle(path + '/' + filename)

df_data = pd.concat([df_train_all, df_test_all], axis=0)

name_prefix = '_FAR.'
name_material = 'Granite'

df_data = df_data[df_data['material'] == name_prefix + name_material]
df_data = df_data.reset_index(drop=True)

# round YIELD and DEPTH to 2 decimal places
df_data.YIELD = df_data.YIELD.round(2)
df_data.DEPTH = df_data.DEPTH.round(2)

df_data.head()


# plot numpy arrays with the DATA column from data_df where YIELD and DEPTH are iterated over from wh_combinations
ws = [1.0]
hs = [100.0]
wh_combinations = list(itertools.product(ws, hs))
out = [[],[],[],[],[],[]]

for w,h in wh_combinations:
    result = compute_partial_derivatives(w=w, h=h, device='cuda', plot=False)
    out[0].append(result['S_t'])
    out[2].append(result['w'])
    out[3].append(result['h'])

    # find the corresponding YIELD and DEPTH values from the dataframe
    idx = df_data[(df_data['YIELD'] == w) & (df_data['DEPTH'] == h)].index[0]
    out[1].append(df_data.loc[idx, 'DATA'])

    

# plot the S(t) and DATA values for each (W,h) combination
fig, axs = plt.subplots(1, len(wh_combinations), figsize=(10, 6), sharex=True, sharey=False)
plt.grid(True)
for i in range(len(wh_combinations)):
    axs.plot(out[0][i], label=f'Python S(t)', color='tab:blue')
    axs.tick_params(axis='y', labelcolor='tab:blue')
    axs.set_title(f'w: {out[2][i]}, h: {out[3][i]}')
    axs.set_ylabel('Python S(t)', color='tab:blue')
    # make a second axis for the DATA values
    ax2 = axs.twinx()
    ax2.plot(out[1][i][98:], label=f'Fortran S(t)', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylabel('Fortran S(t)', color='tab:orange', rotation=270)
    
    
    
    

    # Combine legends
lines_1, labels_1 = axs.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
axs.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
plt.savefig('S_t_comparison.png')


# plot numpy arrays with the DATA column from data_df where YIELD and DEPTH are iterated over from wh_combinations
ws = [1.0]
hs = [100.0]
wh_combinations = list(itertools.product(ws, hs))
out = [[],[],[],[],[],[]]

for w,h in wh_combinations:
    result = compute_partial_derivatives(w=w, h=h, device='cuda', plot=False)
    out[0].append(result['f_t'])
    out[2].append(result['w'])
    out[3].append(result['h'])

    # find the corresponding YIELD and DEPTH values from the dataframe
    idx = df_data[(df_data['YIELD'] == w) & (df_data['DEPTH'] == h)].index[0]
    out[1].append(df_data.loc[idx, 'DATA'])

    

# plot the S(t) and DATA values for each (W,h) combination
fig, axs = plt.subplots(1, len(wh_combinations), figsize=(10, 6), sharex=True, sharey=False)
plt.grid(True)
for i in range(len(wh_combinations)):
    axs.plot(out[0][i], label=f'Python f(t)', color='tab:blue')
    axs.tick_params(axis='y', labelcolor='tab:blue')
    axs.set_title(f'w: {out[2][i]}, h: {out[3][i]}')
    axs.set_ylabel('Python f(t)', color='tab:blue')
    # make a second axis for the DATA values
    ax2 = axs.twinx()
    ax2.plot(out[1][i][98:], label=f'Fortran S(t)', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylabel('Fortran S(t)', color='tab:orange', rotation=270)
    
    
    
    

    # Combine legends
lines_1, labels_1 = axs.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
axs.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
plt.savefig('S_t_vs_f_t_comparison.png')


# plot numpy arrays with the DATA column from data_df where YIELD and DEPTH are iterated over from wh_combinations
ws = [1.0]
hs = [100.0]
wh_combinations = list(itertools.product(ws, hs))
out = [[],[],[],[],[],[]]

for w,h in wh_combinations:
    result = compute_partial_derivatives(w=w, h=h, device='cuda', plot=False)
    out[0].append(result['dS_t_dRel'])
    out[2].append(result['w'])
    out[3].append(result['h'])

    # find the corresponding YIELD and DEPTH values from the dataframe
    idx = df_data[(df_data['YIELD'] == w) & (df_data['DEPTH'] == h)].index[0]
    out[1].append(df_data.loc[idx, 'DATA'])

    

# plot the S(t) and DATA values for each (W,h) combination
fig, axs = plt.subplots(1, len(wh_combinations), figsize=(10, 6), sharex=True, sharey=False)
plt.grid(True)
for i in range(len(wh_combinations)):
    axs.plot(out[0][i], label=f'Python dS(t)/dRel', color='tab:blue')
    axs.tick_params(axis='y', labelcolor='tab:blue')
    axs.set_title(f'w: {out[2][i]}, h: {out[3][i]}')
    axs.set_ylabel('Python dS(t)/dRel', color='tab:blue')
    # make a second axis for the DATA values
    ax2 = axs.twinx()
    ax2.plot(out[1][i][98:], label=f'Fortran S(t)', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylabel('Fortran S(t)', color='tab:orange', rotation=270)
    
    
    
    

    # Combine legends
lines_1, labels_1 = axs.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
axs.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
plt.savefig('S_t_vs_dSdRel_comparison.png')