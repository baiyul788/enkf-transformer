import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

try:
    from scipy.interpolate import interp1d
    from scipy.ndimage import uniform_filter1d
except ImportError:
    interp1d = None
    uniform_filter1d = None

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'

plt.rcParams['backend'] = 'Agg'

def plot_individual_subplots(config, results_data, save_dir=None):
    xTrue = results_data['xTrue']
    xa_enkf = results_data['xa_enkf']
    xa_eakf = results_data.get('xa_eakf', None)
    xa_enkf_tr = results_data['xa_enkf_tr']
    xa_eakf_tr = results_data.get('xa_eakf_tr', None)
    yo = results_data.get('yo', None)
    
    error_enkf = xa_enkf - xTrue
    error_eakf = xa_eakf - xTrue if xa_eakf is not None else None
    xTrue_ds = xTrue[:, ::config.k]
    error_enkf_tr = xa_enkf_tr - xTrue_ds
    error_eakf_tr = xa_eakf_tr - xTrue_ds if xa_eakf_tr is not None else None
    
    time_ds = np.arange(0, xTrue_ds.shape[1]) * config.k * config.dt
    
    if save_dir is None:
        save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    
    saved_paths = []
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3), dpi=300)
    im = ax.imshow(xTrue, aspect='auto', origin='lower',
                   extent=[0, config.tm, 0, max(0, config.n-1)],
                   cmap='turbo', vmin=-15, vmax=15, interpolation='bilinear')
    ax.set_xlabel('Lorenz-96 time', fontsize=8)
    ax.set_ylabel('N(i)', fontsize=8)
    ax.set_xticks(np.arange(0, config.tm+1, 1))
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30, 35])
    ax.tick_params(axis='both', labelsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Value', fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    true_path = os.path.join(save_dir, f'true_tm{config.tm}_seed{config.seed}.png')
    plt.savefig(true_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    saved_paths.append(true_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    if yo is not None:
        obs_2d = np.full((config.n, yo.shape[1]), np.nan)
        di = 4
        obs_indices = np.arange(di-1, config.n, di)
        for i, obs_idx in enumerate(obs_indices):
            if i < yo.shape[0] and obs_idx < config.n:
                obs_2d[obs_idx, :] = yo[i, :]
        
        im = ax.imshow(obs_2d, aspect='auto', origin='lower',
                      extent=[0, config.tm, 1, config.n],
                      cmap='RdYlBu_r', vmin=-15, vmax=15, interpolation='nearest')
    else:
        ax.text(0.5, 0.5, 'No Obs Data', ha='center', va='center', 
                transform=ax.transAxes, )
    
    ax.set_title('Obs', fontsize=8, pad=15)
    ax.set_xlabel('Lorenz-96 time', fontsize=8)
    ax.set_ylabel('N(i)', fontsize=8)
    ax.set_xticks(np.arange(0, config.tm+1, 2))
    ax.set_yticks(np.arange(1, config.n+1, 5))
    ax.tick_params(axis='both', which='major')
    
    if yo is not None:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Value', fontsize=8)
        cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    obs_path = os.path.join(save_dir, f'obs_tm{config.tm}_seed{config.seed}.png')
    plt.savefig(obs_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    saved_paths.append(obs_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(xa_enkf, aspect='auto', origin='lower',
                   extent=[0, config.tm, 1, config.n],
                   cmap='RdYlBu_r', vmin=-15, vmax=15, interpolation='bilinear')
    ax.set_title('EnKF', fontsize=8, pad=15)
    ax.set_xlabel('Lorenz-96 time', fontsize=8)
    ax.set_ylabel('N(i)', fontsize=8)
    ax.set_xticks(np.arange(0, config.tm+1, 2))
    ax.set_yticks(np.arange(1, config.n+1, 5))
    ax.tick_params(axis='both', which='major')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Value', fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    enkf_path = os.path.join(save_dir, f'enkf_tm{config.tm}_seed{config.seed}.png')
    plt.savefig(enkf_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    saved_paths.append(enkf_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(error_enkf, aspect='auto', origin='lower',
                   extent=[0, config.tm, 1, config.n],
                   cmap='RdBu_r', vmin=-15, vmax=15, interpolation='bilinear')
    ax.set_title('EnKF-Error', fontsize=8, pad=15)
    ax.set_xlabel('Lorenz-96 time', fontsize=8)
    ax.set_ylabel('N(i)', fontsize=8)
    ax.set_xticks(np.arange(0, config.tm+1, 2))
    ax.set_yticks(np.arange(1, config.n+1, 5))
    ax.tick_params(axis='both', which='major')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Value', fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    enkf_error_path = os.path.join(save_dir, f'enkf_error_tm{config.tm}_seed{config.seed}.png')
    plt.savefig(enkf_error_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    saved_paths.append(enkf_error_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    time_ds_start = time_ds[0] if len(time_ds) > 0 else 0
    time_ds_end = time_ds[-1] if len(time_ds) > 0 else config.tm
    im = ax.imshow(xa_enkf_tr, aspect='auto', origin='lower',
                   extent=[time_ds_start, time_ds_end, 1, config.n],
                   cmap='RdYlBu_r', vmin=-15, vmax=15, interpolation='bilinear')
    ax.set_title('EnKF+Tr', fontsize=8, pad=15)
    ax.set_xlabel('Lorenz-96 time', fontsize=8)
    ax.set_ylabel('N(i)', fontsize=8)
    ax.set_xticks(np.arange(0, config.tm+1, 2))
    ax.set_yticks(np.arange(1, config.n+1, 5))
    ax.tick_params(axis='both', which='major')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Value', fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    enkf_tr_path = os.path.join(save_dir, f'enkf_tr_tm{config.tm}_seed{config.seed}.png')
    plt.savefig(enkf_tr_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    saved_paths.append(enkf_tr_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(error_enkf_tr, aspect='auto', origin='lower',
                   extent=[time_ds_start, time_ds_end, 1, config.n],
                   cmap='RdBu_r', vmin=-15, vmax=15, interpolation='bilinear')
    ax.set_title('EnKF+Tr-Error', fontsize=8, pad=15)
    ax.set_xlabel('Lorenz-96 time', fontsize=8)
    ax.set_ylabel('N(i)', fontsize=8)
    ax.set_xticks(np.arange(0, config.tm+1, 2))
    ax.set_yticks(np.arange(1, config.n+1, 5))
    ax.tick_params(axis='both', which='major')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Value', fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    enkf_tr_error_path = os.path.join(save_dir, f'enkf_tr_error_tm{config.tm}_seed{config.seed}.png')
    plt.savefig(enkf_tr_error_path, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    saved_paths.append(enkf_tr_error_path)
    
    if xa_eakf is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.imshow(xa_eakf, aspect='auto', origin='lower',
                       extent=[0, config.tm, 1, config.n],
                       cmap='RdYlBu_r', vmin=-15, vmax=15, interpolation='bilinear')
        ax.set_title('EAKF', fontsize=8, pad=15)
        ax.set_xlabel('Lorenz-96 time', fontsize=8)
        ax.set_ylabel('N(i)', fontsize=8)
        ax.set_xticks(np.arange(0, config.tm+1, 2))
        ax.set_yticks(np.arange(1, config.n+1, 5))
        ax.tick_params(axis='both', which='major')
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Value', fontsize=8)
        cbar.ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        eakf_path = os.path.join(save_dir, f'eakf_tm{config.tm}_seed{config.seed}.png')
        plt.savefig(eakf_path, dpi=300, bbox_inches='tight', format='png')
        plt.close(fig)
        saved_paths.append(eakf_path)
    
    if error_eakf is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.imshow(error_eakf, aspect='auto', origin='lower',
                       extent=[0, config.tm, 1, config.n],
                       cmap='RdBu_r', vmin=-15, vmax=15, interpolation='bilinear')
        ax.set_title('EAKF-Error', fontsize=8, pad=15)
        ax.set_xlabel('Lorenz-96 time', fontsize=8)
        ax.set_ylabel('N(i)', fontsize=8)
        ax.set_xticks(np.arange(0, config.tm+1, 2))
        ax.set_yticks(np.arange(1, config.n+1, 5))
        ax.tick_params(axis='both', which='major')
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Value', fontsize=8)
        cbar.ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        eakf_error_path = os.path.join(save_dir, f'eakf_error_tm{config.tm}_seed{config.seed}.png')
        plt.savefig(eakf_error_path, dpi=300, bbox_inches='tight', format='png')
        plt.close(fig)
        saved_paths.append(eakf_error_path)
    
    if xa_eakf_tr is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.imshow(xa_eakf_tr, aspect='auto', origin='lower',
                       extent=[time_ds_start, time_ds_end, 1, config.n],
                       cmap='RdYlBu_r', vmin=-15, vmax=15, interpolation='bilinear')
        ax.set_title('EAKF+Tr', fontsize=8, pad=15)
        ax.set_xlabel('Lorenz-96 time', fontsize=8)
        ax.set_ylabel('N(i)', fontsize=8)
        ax.set_xticks(np.arange(0, config.tm+1, 2))
        ax.set_yticks(np.arange(1, config.n+1, 5))
        ax.tick_params(axis='both', which='major')
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Value', fontsize=8)
        cbar.ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        eakf_tr_path = os.path.join(save_dir, f'eakf_tr_tm{config.tm}_seed{config.seed}.png')
        plt.savefig(eakf_tr_path, dpi=300, bbox_inches='tight', format='png')
        plt.close(fig)
        saved_paths.append(eakf_tr_path)
    
    if error_eakf_tr is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.imshow(error_eakf_tr, aspect='auto', origin='lower',
                       extent=[time_ds_start, time_ds_end, 1, config.n],
                       cmap='RdBu_r', vmin=-15, vmax=15, interpolation='bilinear')
        ax.set_title('EAKF+Tr-Error', fontsize=8, pad=15)
        ax.set_xlabel('Lorenz-96 time', fontsize=8)
        ax.set_ylabel('N(i)', fontsize=8)
        ax.set_xticks(np.arange(0, config.tm+1, 2))
        ax.set_yticks(np.arange(1, config.n+1, 5))
        ax.tick_params(axis='both', which='major')
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Value', fontsize=8)
        cbar.ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        eakf_tr_error_path = os.path.join(save_dir, f'eakf_tr_error_tm{config.tm}_seed{config.seed}.png')
        plt.savefig(eakf_tr_error_path, dpi=300, bbox_inches='tight', format='png')
        plt.close(fig)
        saved_paths.append(eakf_tr_error_path)
    
    return saved_paths


def plot_methods_comparison_grid(config, results_data, save_dir=None, method='enkf'):
    xTrue = results_data['xTrue']
    yo = results_data.get('yo', None)
    
    if method.lower() == 'enkf':
        xa_method = results_data['xa_enkf']
        xa_tr = results_data['xa_enkf_tr']
        method_name = 'EnKF'
    elif method.lower() == 'eakf':
        xa_method = results_data.get('xa_eakf', results_data['xa_enkf'])
        xa_tr = results_data.get('xa_eakf_tr', results_data['xa_enkf_tr'])
        method_name = 'EAKF'
    else:
        raise ValueError(f"method must be 'enkf' or 'eakf', got '{method}'")
    
    error_method = xa_method - xTrue
    xTrue_ds = xTrue[:, ::config.k]
    error_tr = xa_tr - xTrue_ds
    
    obs_2d = None
    if yo is not None:
        obs_2d = np.full((config.n, yo.shape[1]), np.nan)
        di = 4
        obs_indices = np.arange(di-1, config.n, di)
        for i, obs_idx in enumerate(obs_indices):
            if i < yo.shape[0] and obs_idx < config.n:
                obs_2d[obs_idx, :] = yo[i, :]
    
    if save_dir is None:
        save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 12))
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    fontsize_title = 14
    fontsize_label = 13
    fontsize_tick = 11
    fontsize_cbar_label = 12
    fontsize_cbar_tick = 11
    fontsize_note = 12
    
    subplot_width = 0.38
    subplot_height = 0.26
    h_spacing = 0.08
    v_spacing = 0.06
    left_margin = 0.08
    bottom_margin = 0.08
    cbar_width = 0.015
    cbar_gap = 0.01
    
    positions = [
        [left_margin, bottom_margin + 2*(subplot_height + v_spacing), subplot_width, subplot_height],
        [left_margin + subplot_width + h_spacing, bottom_margin + 2*(subplot_height + v_spacing), subplot_width, subplot_height],
        [left_margin, bottom_margin + (subplot_height + v_spacing), subplot_width, subplot_height],
        [left_margin + subplot_width + h_spacing, bottom_margin + (subplot_height + v_spacing), subplot_width, subplot_height],
        [left_margin, bottom_margin, subplot_width, subplot_height],
        [left_margin + subplot_width + h_spacing, bottom_margin, subplot_width, subplot_height],
    ]
    
    vmin_state = -15
    vmax_state = 15
    vmin_error = -15
    vmax_error = 15
    
    state_cmap = 'RdYlBu_r'
    error_cmap = 'RdYlBu_r'
    
    ax1 = fig.add_axes(positions[0])
    im1 = ax1.imshow(xTrue, aspect='auto', origin='lower',
                     extent=[0, config.tm, 1, config.n],
                     cmap=state_cmap, vmin=vmin_state, vmax=vmax_state, interpolation='bilinear')
    ax1.set_title('(a) True', fontfamily='Times New Roman', fontsize=fontsize_title, fontweight='bold', loc='left')
    ax1.set_ylabel('N(i)', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
    ax1.set_xticks(np.arange(0, config.tm+1, 2))
    ax1.set_yticks([1, 10, 20, 30])
    ax1.tick_params(axis='both', which='major', labelsize=fontsize_tick, width=1.0, length=4)
    ax1.tick_params(axis='both', which='minor', width=0.5, length=2)
    
    cax1 = fig.add_axes([positions[0][0] + positions[0][2] + cbar_gap, 
                         positions[0][1], cbar_width, positions[0][3]])
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.ax.tick_params(labelsize=fontsize_cbar_tick)
    cbar1.set_label('Value', fontsize=fontsize_cbar_label, fontweight='normal', fontfamily='Times New Roman')
    
    ax2 = fig.add_axes(positions[1])
    if obs_2d is not None:
        ax2.set_facecolor('white')
        
        obs_times = []
        obs_positions = []
        obs_values = []
        
        for i in range(obs_2d.shape[0]):
            for j in range(obs_2d.shape[1]):
                if not np.isnan(obs_2d[i, j]):
                    obs_times.append(j * config.dt_m)
                    obs_positions.append(i + 1)
                    obs_values.append(obs_2d[i, j])
        
        if len(obs_times) > 0:
            im2 = ax2.scatter(obs_times, obs_positions, c=obs_values, 
                            s=10,
                            cmap=state_cmap, vmin=vmin_state, vmax=vmax_state,
                            marker='o', edgecolors='none', alpha=0.9)
        else:
            im2 = None
    else:
        ax2.text(0.5, 0.5, 'No Obs Data', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=fontsize_note)
        im2 = None
    
    ax2.set_title('(b) Obs', fontfamily='Times New Roman', fontsize=fontsize_title, fontweight='bold', loc='left')
    ax2.set_xlim(0, config.tm)
    ax2.set_ylim(1, config.n)
    ax2.set_xticks(np.arange(0, config.tm+1, 2))
    ax2.set_yticks([1, 10, 20, 30])
    ax2.tick_params(axis='both', which='major', labelsize=fontsize_tick, width=1.0, length=4)
    ax2.tick_params(axis='both', which='minor', width=0.5, length=2)
    
    if im2 is not None:
        cax2 = fig.add_axes([positions[1][0] + positions[1][2] + cbar_gap,
                             positions[1][1], cbar_width, positions[1][3]])
        cbar2 = plt.colorbar(im2, cax=cax2)
        cbar2.ax.tick_params(labelsize=fontsize_cbar_tick)
        cbar2.set_label('Value', fontsize=fontsize_cbar_label, fontweight='normal', fontfamily='Times New Roman')
    
    ax3 = fig.add_axes(positions[2])
    im3 = ax3.imshow(xa_method, aspect='auto', origin='lower',
                     extent=[0, config.tm, 1, config.n],
                     cmap=state_cmap, vmin=vmin_state, vmax=vmax_state, interpolation='bilinear')
    ax3.set_title(f'(c) {method_name}', fontfamily='Times New Roman', fontsize=fontsize_title, fontweight='bold', loc='left')
    ax3.set_ylabel('N(i)', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
    ax3.set_xticks(np.arange(0, config.tm+1, 2))
    ax3.set_yticks([1, 10, 20, 30])
    ax3.tick_params(axis='both', which='major', labelsize=fontsize_tick, width=1.0, length=4)
    ax3.tick_params(axis='both', which='minor', width=0.5, length=2)
    
    cax3 = fig.add_axes([positions[2][0] + positions[2][2] + cbar_gap,
                         positions[2][1], cbar_width, positions[2][3]])
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar3.ax.tick_params(labelsize=fontsize_cbar_tick)
    cbar3.set_label('Value', fontsize=fontsize_cbar_label, fontweight='normal', fontfamily='Times New Roman')
    
    ax4 = fig.add_axes(positions[3])
    im4 = ax4.imshow(error_method, aspect='auto', origin='lower',
                     extent=[0, config.tm, 1, config.n],
                     cmap=error_cmap, vmin=vmin_error, vmax=vmax_error, interpolation='bilinear')
    ax4.set_title(f'(d) {method_name}-Error', fontfamily='Times New Roman', fontsize=fontsize_title, fontweight='bold', loc='left')
    ax4.set_xticks(np.arange(0, config.tm+1, 2))
    ax4.set_yticks([1, 10, 20, 30])
    ax4.tick_params(axis='both', which='major', labelsize=fontsize_tick, width=1.0, length=4)
    ax4.tick_params(axis='both', which='minor', width=0.5, length=2)
    
    cax4 = fig.add_axes([positions[3][0] + positions[3][2] + cbar_gap,
                         positions[3][1], cbar_width, positions[3][3]])
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar4.ax.tick_params(labelsize=fontsize_cbar_tick)
    cbar4.set_label('Error', fontsize=fontsize_cbar_label, fontweight='normal', fontfamily='Times New Roman')
    
    ax5 = fig.add_axes(positions[4])
    time_ds_start = 0
    time_ds_end = (xa_tr.shape[1] - 1) * config.k * config.dt
    im5 = ax5.imshow(xa_tr, aspect='auto', origin='lower',
                     extent=[time_ds_start, time_ds_end, 1, config.n],
                     cmap=state_cmap, vmin=vmin_state, vmax=vmax_state, interpolation='bilinear')
    ax5.set_title(f'(e) {method_name}-Transformer', fontfamily='Times New Roman', fontsize=fontsize_title, fontweight='bold', loc='left')
    ax5.set_xlabel('Lorenz-96 time', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
    ax5.set_ylabel('N(i)', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
    ax5.set_xticks(np.arange(0, config.tm+1, 2))
    ax5.set_yticks([1, 10, 20, 30])
    ax5.tick_params(axis='both', which='major', labelsize=fontsize_tick, width=1.0, length=4)
    ax5.tick_params(axis='both', which='minor', width=0.5, length=2)
    
    cax5 = fig.add_axes([positions[4][0] + positions[4][2] + cbar_gap,
                         positions[4][1], cbar_width, positions[4][3]])
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar5.ax.tick_params(labelsize=fontsize_cbar_tick)
    cbar5.set_label('Value', fontsize=fontsize_cbar_label, fontweight='normal', fontfamily='Times New Roman')
    
    ax6 = fig.add_axes(positions[5])
    im6 = ax6.imshow(error_tr, aspect='auto', origin='lower',
                     extent=[time_ds_start, time_ds_end, 1, config.n],
                     cmap=error_cmap, vmin=vmin_error, vmax=vmax_error, interpolation='bilinear')
    ax6.set_title(f'(f) {method_name}-Transformer-Error', fontfamily='Times New Roman', fontsize=fontsize_title, fontweight='bold', loc='left')
    ax6.set_xlabel('Lorenz-96 time', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
    ax6.set_xticks(np.arange(0, config.tm+1, 2))
    ax6.set_yticks([1, 10, 20, 30])
    ax6.tick_params(axis='both', which='major', labelsize=fontsize_tick, width=1.0, length=4)
    ax6.tick_params(axis='both', which='minor', width=0.5, length=2)
    
    cax6 = fig.add_axes([positions[5][0] + positions[5][2] + cbar_gap,
                         positions[5][1], cbar_width, positions[5][3]])
    cbar6 = plt.colorbar(im6, cax=cax6)
    cbar6.ax.tick_params(labelsize=fontsize_cbar_tick)
    cbar6.set_label('Error', fontsize=fontsize_cbar_label, fontweight='normal', fontfamily='Times New Roman')
    
    save_path = os.path.join(save_dir, 
                            f'figure4_{method.lower()}_comparison_grid_N{config.N}_s{config.s}_tm{config.tm}_k{config.k}_sp{config.sig_p}_sm{config.sig_m}_seed{config.seed}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', format='png')
    plt.close('all')
    
    return save_path


def plot_error_and_uncertainty_analysis(config, xTrue, xa_enkf, xa_eakf, xa_enkf_tr, xa_eakf_tr,
                                       xb_enkf=None, xb_eakf=None, xb_enkf_tr=None, xb_eakf_tr=None,
                                       variable_idx=19, save_dir=None):
    
    if save_dir is None:
        save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 14))
    fontsize_title = 14
    fontsize_label = 13
    fontsize_tick = 11
    fontsize_legend = 11
    fontsize_value = 11

    plt.rcParams['ytick.labelsize'] = fontsize_tick
    plt.rcParams['legend.fontsize'] = fontsize_legend
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'bold'
    
    error_width = 0.43
    error_height = 0.17
    h_spacing = 0.06
    v_spacing = 0.07
    v_spacing_large = 0.08
    left_margin = 0.08
    bottom_margin = 0.08
    
    ci_height = 0.17
    
    error_positions = [
        [left_margin, bottom_margin + ci_height + v_spacing_large + v_spacing + error_height, error_width, error_height],
        [left_margin + error_width + h_spacing, bottom_margin + ci_height + v_spacing_large + v_spacing + error_height, error_width, error_height],
        [left_margin, bottom_margin + ci_height + v_spacing_large, error_width, error_height],
        [left_margin + error_width + h_spacing, bottom_margin + ci_height + v_spacing_large, error_width, error_height],
    ]
    
    ci_position = [left_margin, bottom_margin, error_width*2 + h_spacing, ci_height]
    
    time_full = np.arange(xTrue.shape[1]) * config.dt
    time_ds = np.arange(xa_enkf_tr.shape[1]) * config.k * config.dt
    
    true_var = xTrue[variable_idx, :]
    mean_enkf = xa_enkf[variable_idx, :]
    mean_eakf = xa_eakf[variable_idx, :]
    
    true_var_ds = true_var[::config.k][:xa_enkf_tr.shape[1]]
    mean_enkf_tr = xa_enkf_tr[variable_idx, :]
    mean_eakf_tr = xa_eakf_tr[variable_idx, :]
    
    all_values = np.concatenate([true_var, mean_enkf, mean_eakf, 
                                 true_var_ds, mean_enkf_tr, mean_eakf_tr])
    value_range = np.max(all_values) - np.min(all_values)
    y_min = np.min(all_values) - 0.1 * value_range
    y_max = np.max(all_values) + 0.1 * value_range
    
    colors = {
        'enkf': '#D32F2F',
        'enkf_tr': '#1976D2',
        'eakf': '#F57C00',
        'eakf_tr': '#388E3C',
    }
    
    ax1 = fig.add_axes(error_positions[0])
    ax1.set_facecolor('white')
    
    if xb_enkf is not None:
        ensemble_var = xb_enkf[:, variable_idx, :]
        ci_lower_enkf = np.percentile(ensemble_var, 5, axis=0)
        ci_upper_enkf = np.percentile(ensemble_var, 95, axis=0)
    else:
        error = mean_enkf - true_var
        window_size = max(10, len(error) // 20)
        if uniform_filter1d is not None:
            error_squared = error ** 2
            error_var_smooth = uniform_filter1d(error_squared, size=window_size, mode='nearest')
            error_std_smooth = np.sqrt(error_var_smooth)
        else:
            error_std_smooth = np.full_like(error, np.std(error))
        ci_lower_enkf = mean_enkf - 1.645 * error_std_smooth
        ci_upper_enkf = mean_enkf + 1.645 * error_std_smooth
    
    ax1.fill_between(time_full, ci_lower_enkf, ci_upper_enkf, 
                     color=colors['enkf'], alpha=0.15, label='90% interval', zorder=1, edgecolor='none')
    ax1.plot(time_full, ci_lower_enkf, color=colors['enkf'], linestyle='--', 
             linewidth=1.0, alpha=0.6, zorder=2, dashes=(5, 3))
    ax1.plot(time_full, ci_upper_enkf, color=colors['enkf'], linestyle='--', 
             linewidth=1.0, alpha=0.6, zorder=2, dashes=(5, 3))
    ax1.plot(time_full, mean_enkf, color=colors['enkf'], linewidth=1.5, 
             label='Mean', alpha=1.0, zorder=3)
    ax1.plot(time_full, true_var, color='#FF6B35', linewidth=1.5, 
             label='True', alpha=1.0, zorder=4, linestyle='-')
    ax1.set_title(f'(a) EnKF - Variable x({variable_idx})', fontfamily='Times New Roman', fontsize=fontsize_title, fontweight='bold', loc='left', pad=8)
    ax1.set_xlabel('Lorenz-96 time', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
    ax1.set_ylabel('N(i)', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
    ax1.set_xlim(0, config.tm)
    ax1.set_ylim(y_min, y_max)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize_tick, width=1.0, length=4)
    for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontsize(fontsize_tick)
    ax1.legend(loc='upper left', fontsize=fontsize_legend, framealpha=0.9, edgecolor='gray', fancybox=False, frameon=True)
    ax1.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    ax2 = fig.add_axes(error_positions[1])
    ax2.set_facecolor('white')
    
    if xb_enkf_tr is not None:
        ensemble_var_tr = xb_enkf_tr[:, variable_idx, :]
        ci_lower_enkf_tr = np.percentile(ensemble_var_tr, 5, axis=0)
        ci_upper_enkf_tr = np.percentile(ensemble_var_tr, 95, axis=0)
    else:
        error_tr = mean_enkf_tr - true_var_ds
        window_size = max(5, len(error_tr) // 10)
        if uniform_filter1d is not None:
            error_squared_tr = error_tr ** 2
            error_var_smooth_tr = uniform_filter1d(error_squared_tr, size=window_size, mode='nearest')
            error_std_smooth_tr = np.sqrt(error_var_smooth_tr)
        else:
            error_std_smooth_tr = np.full_like(error_tr, np.std(error_tr))
        ci_lower_enkf_tr = mean_enkf_tr - 1.645 * error_std_smooth_tr
        ci_upper_enkf_tr = mean_enkf_tr + 1.645 * error_std_smooth_tr
    
    ax2.fill_between(time_ds, ci_lower_enkf_tr, ci_upper_enkf_tr, 
                     color=colors['enkf_tr'], alpha=0.15, label='90% interval', zorder=1, edgecolor='none')
    ax2.plot(time_ds, ci_lower_enkf_tr, color=colors['enkf_tr'], linestyle='--', 
             linewidth=1.0, alpha=0.6, zorder=2, dashes=(5, 3))
    ax2.plot(time_ds, ci_upper_enkf_tr, color=colors['enkf_tr'], linestyle='--', 
             linewidth=1.0, alpha=0.6, zorder=2, dashes=(5, 3))
    ax2.plot(time_ds, mean_enkf_tr, color=colors['enkf_tr'], linewidth=1.5, 
             label='Mean', alpha=1.0, zorder=3)
    ax2.plot(time_ds, true_var_ds, color='#FF6B35', linewidth=1.5, 
             label='True', alpha=1.0, zorder=4, linestyle='-')
    ax2.set_title(f'(b) EnKF-Transformer - Variable x({variable_idx})', fontfamily='Times New Roman', fontsize=fontsize_title, fontweight='bold', loc='left', pad=8)
    ax2.set_xlabel('Lorenz-96 time', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
    ax2.set_xlim(0, config.tm)
    ax2.set_ylim(y_min, y_max)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize_tick, width=1.0, length=4)
    for tick in ax2.get_xticklabels() + ax2.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontsize(fontsize_tick)
    ax2.legend(loc='upper left', fontsize=fontsize_legend, framealpha=0.9, edgecolor='gray', fancybox=False, frameon=True)
    ax2.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    ax3 = fig.add_axes(error_positions[2])
    ax3.set_facecolor('white')
    
    if xb_eakf is not None:
        ensemble_var_eakf = xb_eakf[:, variable_idx, :]
        ci_lower_eakf = np.percentile(ensemble_var_eakf, 5, axis=0)
        ci_upper_eakf = np.percentile(ensemble_var_eakf, 95, axis=0)
    else:
        error_eakf = mean_eakf - true_var
        window_size = max(10, len(error_eakf) // 20)
        if uniform_filter1d is not None:
            error_squared_eakf = error_eakf ** 2
            error_var_smooth_eakf = uniform_filter1d(error_squared_eakf, size=window_size, mode='nearest')
            error_std_smooth_eakf = np.sqrt(error_var_smooth_eakf)
        else:
            error_std_smooth_eakf = np.full_like(error_eakf, np.std(error_eakf))
        ci_lower_eakf = mean_eakf - 1.645 * error_std_smooth_eakf
        ci_upper_eakf = mean_eakf + 1.645 * error_std_smooth_eakf
    
    ax3.fill_between(time_full, ci_lower_eakf, ci_upper_eakf, 
                     color=colors['eakf'], alpha=0.15, label='90% interval', zorder=1, edgecolor='none')
    ax3.plot(time_full, ci_lower_eakf, color=colors['eakf'], linestyle='--', 
             linewidth=1.0, alpha=0.6, zorder=2, dashes=(5, 3))
    ax3.plot(time_full, ci_upper_eakf, color=colors['eakf'], linestyle='--', 
             linewidth=1.0, alpha=0.6, zorder=2, dashes=(5, 3))
    ax3.plot(time_full, mean_eakf, color=colors['eakf'], linewidth=1.5, 
             label='Mean', alpha=1.0, zorder=3)
    ax3.plot(time_full, true_var, color='#FF6B35', linewidth=1.5, 
             label='True', alpha=1.0, zorder=4, linestyle='-')
    ax3.set_title(f'(c) EAKF - Variable x({variable_idx})', fontfamily='Times New Roman', fontsize=fontsize_title, fontweight='bold', loc='left', pad=8)
    ax3.set_xlabel('Lorenz-96 time', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
    ax3.set_ylabel('N(i)', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
    ax3.set_xlim(0, config.tm)
    ax3.set_ylim(y_min, y_max)
    ax3.tick_params(axis='both', which='major', labelsize=fontsize_tick, width=1.0, length=4)
    for tick in ax3.get_xticklabels() + ax3.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontsize(fontsize_tick)
    ax3.legend(loc='upper left', fontsize=fontsize_legend, framealpha=0.9, edgecolor='gray', fancybox=False, frameon=True)
    ax3.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    ax4 = fig.add_axes(error_positions[3])
    ax4.set_facecolor('white')
    
    if xb_eakf_tr is not None:
        ensemble_var_eakf_tr = xb_eakf_tr[:, variable_idx, :]
        ci_lower_eakf_tr = np.percentile(ensemble_var_eakf_tr, 5, axis=0)
        ci_upper_eakf_tr = np.percentile(ensemble_var_eakf_tr, 95, axis=0)
    else:
        error_eakf_tr = mean_eakf_tr - true_var_ds
        window_size = max(5, len(error_eakf_tr) // 10)
        if uniform_filter1d is not None:
            error_squared_eakf_tr = error_eakf_tr ** 2
            error_var_smooth_eakf_tr = uniform_filter1d(error_squared_eakf_tr, size=window_size, mode='nearest')
            error_std_smooth_eakf_tr = np.sqrt(error_var_smooth_eakf_tr)
        else:
            error_std_smooth_eakf_tr = np.full_like(error_eakf_tr, np.std(error_eakf_tr))
        ci_lower_eakf_tr = mean_eakf_tr - 1.645 * error_std_smooth_eakf_tr
        ci_upper_eakf_tr = mean_eakf_tr + 1.645 * error_std_smooth_eakf_tr
    
    ax4.fill_between(time_ds, ci_lower_eakf_tr, ci_upper_eakf_tr, 
                     color=colors['eakf_tr'], alpha=0.15, label='90% interval', zorder=1, edgecolor='none')
    ax4.plot(time_ds, ci_lower_eakf_tr, color=colors['eakf_tr'], linestyle='--', 
             linewidth=1.0, alpha=0.6, zorder=2, dashes=(5, 3))
    ax4.plot(time_ds, ci_upper_eakf_tr, color=colors['eakf_tr'], linestyle='--', 
             linewidth=1.0, alpha=0.6, zorder=2, dashes=(5, 3))
    ax4.plot(time_ds, mean_eakf_tr, color=colors['eakf_tr'], linewidth=1.5, 
             label='Mean', alpha=1.0, zorder=3)
    ax4.plot(time_ds, true_var_ds, color='#FF6B35', linewidth=1.5, 
             label='True', alpha=1.0, zorder=4, linestyle='-')
    ax4.set_title(f'(d) EAKF-Transformer - Variable x({variable_idx})', fontfamily='Times New Roman', fontsize=fontsize_title, fontweight='bold', loc='left', pad=8)
    ax4.set_xlabel('Lorenz-96 time', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
    ax4.set_xlim(0, config.tm)
    ax4.set_ylim(y_min, y_max)
    ax4.tick_params(axis='both', which='major', labelsize=fontsize_tick, width=1.0, length=4)
    for tick in ax4.get_xticklabels() + ax4.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
        tick.set_fontsize(fontsize_tick)
    ax4.legend(loc='upper left', fontsize=fontsize_legend, framealpha=0.9, edgecolor='gray', fancybox=False, frameon=True)
    ax4.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    ax5 = fig.add_axes(ci_position)
    
    if xb_enkf is not None and xb_eakf is not None:
        std_enkf = 2 * np.std(xb_enkf[:, variable_idx, :], axis=0)
        std_eakf = 2 * np.std(xb_eakf[:, variable_idx, :], axis=0)
        
        ax5.plot(time_full, std_enkf, color=colors['enkf'], linewidth=1.5, label=f'EnKF (mean={np.mean(std_enkf):.2f})', alpha=0.8)
        ax5.plot(time_full, std_eakf, color=colors['eakf'], linewidth=1.5, label=f'EAKF (mean={np.mean(std_eakf):.2f})', alpha=0.8)
        
        if xb_enkf_tr is not None:
            std_enkf_tr = 2 * np.std(xb_enkf_tr[:, variable_idx, :], axis=0)
            ax5.plot(time_ds, std_enkf_tr, color=colors['enkf_tr'], linewidth=1.5, label=f'EnKF+TR (mean={np.mean(std_enkf_tr):.2f})', alpha=0.8)
        
        if xb_eakf_tr is not None:
            std_eakf_tr = 2 * np.std(xb_eakf_tr[:, variable_idx, :], axis=0)
            ax5.plot(time_ds, std_eakf_tr, color=colors['eakf_tr'], linewidth=1.5, label=f'EAKF+TR (mean={np.mean(std_eakf_tr):.2f})', alpha=0.8)
        
        ax5.set_title(f'(e) Uncertainty Evolution (95% CI Width) - Variable x({variable_idx})', 
                      fontfamily='Times New Roman', fontsize=fontsize_title, fontweight='bold', loc='left')
        ax5.set_ylabel('2σ (CI Width)', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
        ax5.set_xlabel('Lorenz-96 time', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
        ax5.set_xlim(0, config.tm)
        ax5.tick_params(axis='both', which='major', labelsize=fontsize_tick)
        ax5.legend(loc='upper right', fontsize=fontsize_legend, ncol=2, framealpha=0.95, edgecolor='gray', fancybox=True)
        ax5.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    else:
        error_enkf = np.abs(mean_enkf - true_var)
        error_eakf = np.abs(mean_eakf - true_var)
        error_enkf_tr = np.abs(mean_enkf_tr - true_var_ds)
        error_eakf_tr = np.abs(mean_eakf_tr - true_var_ds)
        
        rmse_enkf = np.sqrt(np.mean(error_enkf**2))
        rmse_eakf = np.sqrt(np.mean(error_eakf**2))
        rmse_enkf_tr = np.sqrt(np.mean(error_enkf_tr**2))
        rmse_eakf_tr = np.sqrt(np.mean(error_eakf_tr**2))
        
        mean_error_enkf = np.mean(error_enkf)
        mean_error_eakf = np.mean(error_eakf)
        mean_error_enkf_tr = np.mean(error_enkf_tr)
        mean_error_eakf_tr = np.mean(error_eakf_tr)
        
        max_enkf = np.max(error_enkf)
        max_eakf = np.max(error_eakf)
        max_enkf_tr = np.max(error_enkf_tr)
        max_eakf_tr = np.max(error_eakf_tr)
        
        methods = ['EnKF', 'EAKF', 'EnKF+TR', 'EAKF+TR']
        rmse_values = [rmse_enkf, rmse_eakf, rmse_enkf_tr, rmse_eakf_tr]
        mean_values = [mean_error_enkf, mean_error_eakf, mean_error_enkf_tr, mean_error_eakf_tr]
        max_values = [max_enkf, max_eakf, max_enkf_tr, max_eakf_tr]
        
        rmse_color = '#2E86AB'
        mean_color = '#A23B72'
        max_color = '#F18F01'
        
        x_pos = np.arange(len(methods))
        width = 0.25
        
        bars1 = ax5.bar(x_pos - width, rmse_values, width, label='RMSE', 
                       color=rmse_color, alpha=0.9, edgecolor='white', linewidth=1.2)
        bars2 = ax5.bar(x_pos, mean_values, width, label='Mean Error', 
                       color=mean_color, alpha=0.9, edgecolor='white', linewidth=1.2)
        bars3 = ax5.bar(x_pos + width, max_values, width, label='Max Error', 
                       color=max_color, alpha=0.9, edgecolor='white', linewidth=1.2)
        
        for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            height3 = bar3.get_height()
            ax5.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.1, f'{height1:.2f}', 
                    ha='center', va='bottom', fontsize=fontsize_value, fontfamily='Times New Roman', fontweight='normal')
            ax5.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.1, f'{height2:.2f}', 
                    ha='center', va='bottom', fontsize=fontsize_value, fontfamily='Times New Roman', fontweight='normal')
            ax5.text(bar3.get_x() + bar3.get_width()/2., height3 + 0.1, f'{height3:.2f}', 
                    ha='center', va='bottom', fontsize=fontsize_value, fontfamily='Times New Roman', fontweight='normal')
        
        ax5.set_title(f'(e) Error Statistics Comparison - Variable x({variable_idx})', 
                      fontfamily='Times New Roman', fontsize=fontsize_title, fontweight='bold', loc='left')
        ax5.set_ylabel('Error Magnitude', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
        ax5.set_xlabel('Methods', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(methods, fontsize=fontsize_tick, fontweight='bold')
        ax5.tick_params(axis='y', which='major', labelsize=fontsize_tick)
        ax5.legend(loc='upper left', fontsize=fontsize_legend, framealpha=0.95, edgecolor='gray', fancybox=True)
        ax5.grid(True, axis='y', alpha=0.2, linestyle=':', linewidth=0.5)
        ax5.set_ylim(0, max(max_values) * 1.15)
    
    save_path = os.path.join(save_dir, 
                            f'figure5_error_uncertainty_analysis_x{variable_idx}_N{config.N}_s{config.s}_tm{config.tm}_k{config.k}_sp{config.sig_p}_sm{config.sig_m}_seed{config.seed}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', format='png')
    plt.close('all')
    
    return save_path


def plot_multivar_state_comparison(config, results_data, save_dir=None, method='enkf', variables=None):
    
    xTrue = results_data['xTrue']  # [36, nt+1]
    xa_enkf = results_data['xa_enkf']  # [36, nt+1]
    xa_eakf = results_data.get('xa_eakf', None)  # [36, nt+1]
    xa_enkf_tr = results_data['xa_enkf_tr']  # [36, T_ds]
    xa_eakf_tr = results_data.get('xa_eakf_tr', None)  # [36, T_ds]

    fontsize_title = 14
    fontsize_label = 13
    fontsize_tick = 11
    fontsize_legend = 11
    
    if variables is None:
        variables = [9, 19, 29]
    
    time_full = np.arange(xTrue.shape[1]) * config.dt
    xTrue_ds = xTrue[:, ::config.k]
    time_ds = np.arange(xTrue_ds.shape[1]) * config.k * config.dt
    
    if save_dir is None:
        save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    
    colors = {
        'true': '#000000',
        'enkf': '#D32F2F',
        'eakf': '#FF7043',
        'enkf_tr': '#1976D2',
        'eakf_tr': '#388E3C'
    }
    
    linestyles = {
        'true': '-',
        'enkf': '--',
        'eakf': '--',
        'enkf_tr': '-',
        'eakf_tr': '-'
    }
    
    linewidths = {
        'true': 1.5,
        'enkf': 1.5,
        'eakf': 1.5,
        'enkf_tr': 1.5,
        'eakf_tr': 1.5
    }
    
    n_vars = len(variables)
    fig = plt.figure(figsize=(12, 2.5 * n_vars))
    
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    subplot_labels = subplot_labels[:n_vars]
    var_labels = [f'x({v+1})' for v in variables]
    
    for i, var_idx in enumerate(variables):
        ax = fig.add_subplot(n_vars, 1, i+1)
        
        ax.set_facecolor('white')
        
        ax.plot(time_full, xTrue[var_idx, :], 
                color=colors['true'], linestyle=linestyles['true'], linewidth=linewidths['true'],
                label='True', alpha=0.9, zorder=5)
        
        ax.plot(time_full, xa_enkf[var_idx, :], 
                color=colors['enkf'], linestyle=linestyles['enkf'], linewidth=linewidths['enkf'],
                label='EnKF', alpha=0.8, zorder=3)
        
        if xa_eakf is not None:
            ax.plot(time_full, xa_eakf[var_idx, :], 
                    color=colors['eakf'], linestyle=linestyles['eakf'], linewidth=linewidths['eakf'],
                    label='EAKF', alpha=0.8, zorder=3)
        
        ax.plot(time_ds, xa_enkf_tr[var_idx, :], 
                color=colors['enkf_tr'], linestyle=linestyles['enkf_tr'], linewidth=linewidths['enkf_tr'],
                label='EnKF-Transformer', alpha=0.9, zorder=4)
        
        if xa_eakf_tr is not None:
            ax.plot(time_ds, xa_eakf_tr[var_idx, :], 
                    color=colors['eakf_tr'], linestyle=linestyles['eakf_tr'], linewidth=linewidths['eakf_tr'],
                    label='EAKF-Transformer', alpha=0.95, zorder=4)
        
        ax.set_title(f'{subplot_labels[i]} Variable {var_labels[i]} - State Value Reconstruction', 
                     fontfamily='Times New Roman', fontsize=fontsize_title, fontweight='bold', loc='left', pad=10)
        ax.set_ylabel('State Value', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
        
        if i == len(variables) - 1:
            ax.set_xlabel('Lorenz-96 time', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
        
        ax.set_xlim(0, config.tm)
        ax.tick_params(axis='both', which='major', labelsize=fontsize_tick, width=1.0, length=4)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily('Times New Roman')
            tick.set_fontsize(fontsize_tick)
        
        if i == 0:
            ax.legend(loc='upper left', 
                     fontsize=fontsize_legend, framealpha=0.9, 
                     edgecolor='gray', fancybox=False, shadow=False,
                     ncol=2, handlelength=2.0, labelspacing=0.3,
                     columnspacing=1.0, borderaxespad=0.5, frameon=True)
        
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8, color='gray')
        
        y_min = min(xTrue[var_idx, :].min(), xa_enkf[var_idx, :].min(), xa_enkf_tr[var_idx, :].min())
        y_max = max(xTrue[var_idx, :].max(), xa_enkf[var_idx, :].max(), xa_enkf_tr[var_idx, :].max())
        if xa_eakf is not None:
            y_min = min(y_min, xa_eakf[var_idx, :].min())
            y_max = max(y_max, xa_eakf[var_idx, :].max())
        if xa_eakf_tr is not None:
            y_min = min(y_min, xa_eakf_tr[var_idx, :].min())
            y_max = max(y_max, xa_eakf_tr[var_idx, :].max())
        
        if y_min < 0 < y_max:
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3, zorder=1)
    
    plt.tight_layout(h_pad=2.5)
    
    save_path = os.path.join(save_dir, 
                            f'figure6_multivar_state_comparison_{method}_N{config.N}_s{config.s}_tm{config.tm}_k{config.k}_sp{config.sig_p}_sm{config.sig_m}_seed{config.seed}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2, format='png')
    plt.close('all')
    
    print(f"Figure 6 saved: {save_path}")
    return save_path


def plot_parameter_sensitivity_grid(sensitivity_data_dict, save_dir=None):
    import pandas as pd
    
    if save_dir is None:
        save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    
    colors = {
        'enkf': '#D32F2F',
        'eakf': '#FF7043',
        'enkf_tr': '#1976D2',
        'eakf_tr': '#388E3C'
    }
    
    linestyles = {
        'enkf': '--',
        'eakf': '--',
        'enkf_tr': '-',
        'eakf_tr': '-'
    }
    
    markers = {
        'enkf': 'o',
        'eakf': '^',
        'enkf_tr': 'o',
        'eakf_tr': '^'
    }
    
    linewidths = {
        'enkf': 1.5,
        'eakf': 1.5,
        'enkf_tr': 1.5,
        'eakf_tr': 1.5
    }
    
    markersizes = {
        'enkf': 6,
        'eakf': 6,
        'enkf_tr': 7,
        'eakf_tr': 8
    }
    
    fig = plt.figure(figsize=(14, 10))

    fontsize_title = 14
    fontsize_label = 13
    fontsize_tick = 11
    fontsize_legend = 11
    
    subplot_configs = [
        {
            'pos': (2, 2, 1),
            'data_key': 's_sensitivity',
            'param_name': 's',
            'xlabel': 'Observation Density $s$',
            'title': '(a) Observation Density Sensitivity',
            'param_values': [9, 12, 18]
        },
        {
            'pos': (2, 2, 2),
            'data_key': 'N_sensitivity',
            'param_name': 'N',
            'xlabel': 'Ensemble Size $N$',
            'title': '(b) Ensemble Size Sensitivity',
            'param_values': [30, 50, 70]
        },
        {
            'pos': (2, 2, 3),
            'data_key': 'sig_p_sensitivity',
            'param_name': 'sig_p',
            'xlabel': 'Model Noise $\\sigma_{\\mathrm{p}}$',
            'title': '(c) Model Noise Sensitivity',
            'param_values': [0.45, 0.5, 0.6, 0.65]
        },
        {
            'pos': (2, 2, 4),
            'data_key': 'sig_m_sensitivity',
            'param_name': 'sig_m',
            'xlabel': 'Observation Noise $\\sigma_{\\mathrm{m}}$',
            'title': '(d) Observation Noise Sensitivity',
            'param_values': [0.05, 0.15, 0.3, 0.35, 0.4]
        }
    ]
    
    all_rmse_values = []
    
    for i, config in enumerate(subplot_configs):
        ax = fig.add_subplot(*config['pos'])
        ax.set_facecolor('white')
        
        data_key = config['data_key']
        if data_key not in sensitivity_data_dict:
            continue
        
        df = sensitivity_data_dict[data_key]
        param_name = config['param_name']
        param_values = config['param_values']
        
        rmse_enkf = []
        rmse_eakf = []
        rmse_enkf_tr = []
        rmse_eakf_tr = []
        
        for val in param_values:
            row = df[df['parameter_value'] == val]
            if len(row) > 0:
                rmse_enkf.append(float(row['rmse_enkf'].iloc[0] if hasattr(row['rmse_enkf'], 'iloc') else row['rmse_enkf'][0]))
                rmse_eakf.append(float(row['rmse_eakf'].iloc[0] if hasattr(row['rmse_eakf'], 'iloc') else row['rmse_eakf'][0]))
                rmse_enkf_tr.append(float(row['rmse_enkf_tr'].iloc[0] if hasattr(row['rmse_enkf_tr'], 'iloc') else row['rmse_enkf_tr'][0]))
                rmse_eakf_tr.append(float(row['rmse_eakf_tr'].iloc[0] if hasattr(row['rmse_eakf_tr'], 'iloc') else row['rmse_eakf_tr'][0]))
            else:
                rmse_enkf.append(np.nan)
                rmse_eakf.append(np.nan)
                rmse_enkf_tr.append(np.nan)
                rmse_eakf_tr.append(np.nan)
        
        all_rmse_values.extend([v for v in rmse_enkf + rmse_eakf + rmse_enkf_tr + rmse_eakf_tr if not np.isnan(v)])
        
        ax.plot(param_values, rmse_enkf, 
                color=colors['enkf'], linestyle=linestyles['enkf'], marker=markers['enkf'],
                linewidth=linewidths['enkf'], markersize=markersizes['enkf'], 
                label='EnKF', alpha=0.8, markerfacecolor='white', markeredgewidth=1.5)
        
        ax.plot(param_values, rmse_eakf, 
                color=colors['eakf'], linestyle=linestyles['eakf'], marker=markers['eakf'],
                linewidth=linewidths['eakf'], markersize=markersizes['eakf'], 
                label='EAKF', alpha=0.8, markerfacecolor='white', markeredgewidth=1.5)
        
        ax.plot(param_values, rmse_enkf_tr, 
                color=colors['enkf_tr'], linestyle=linestyles['enkf_tr'], marker=markers['enkf_tr'],
                linewidth=linewidths['enkf_tr'], markersize=markersizes['enkf_tr'], 
                label='EnKF-Transformer', alpha=0.9, markerfacecolor=colors['enkf_tr'])
        
        ax.plot(param_values, rmse_eakf_tr, 
                color=colors['eakf_tr'], linestyle=linestyles['eakf_tr'], marker=markers['eakf_tr'],
                linewidth=linewidths['eakf_tr'], markersize=markersizes['eakf_tr'], 
                label='EAKF-Transformer', alpha=0.95, markerfacecolor=colors['eakf_tr'])
        
        ax.set_title(config['title'], fontfamily='Times New Roman', fontsize=fontsize_title, fontweight='bold', loc='left', pad=10)
        ax.set_xlabel(config['xlabel'], fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
        
        if i % 2 == 0:
            ax.set_ylabel('RMSE', fontfamily='Times New Roman', fontsize=fontsize_label, fontweight='bold')
        
        ax.tick_params(axis='both', which='major', labelsize=fontsize_tick, width=1.0, length=4)
        
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, color='gray')
        
        if i == 0:
            ax.legend(loc='upper left', fontsize=fontsize_legend, framealpha=0.95, 
                     edgecolor='gray', fancybox=True, shadow=True)
    
    if len(all_rmse_values) > 0:
        y_min = min(all_rmse_values)
        y_max = max(all_rmse_values)
        y_range = y_max - y_min
        y_margin = y_range * 0.1
        
        unified_ylim = (max(0, y_min - y_margin), y_max + y_margin)
        
        for i in range(4):
            ax = fig.axes[i]
            ax.set_ylim(unified_ylim)
    
    plt.tight_layout(w_pad=3.0, h_pad=3.0)
    
    save_path = os.path.join(save_dir, 'figure7_parameter_sensitivity_grid.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', format='png')
    plt.close('all')
    
    return save_path

