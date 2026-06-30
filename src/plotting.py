import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, lognorm, weibull_min, rayleigh
from src.calculations import get_strength_dist, get_load_pdf

# Design tokens
COLORS = {
    'vert': '#D85A30',  # Orange-red
    'lat': '#185FA5',   # Muted Blue
    'tors': '#800080',  # Purple
    'combined': '#003366', # Dark blue
    'strength': '#5DCAA5', # Emerald Green
    'bg': '#FFFFFF',
    'spine': '#CCCCCC'
}

def plot_swbm_static(df_swbm, col_name):
    """
    Plots the Still Water Bending Moment along the ship length.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    pos = df_swbm['Position (m)'].values
    moments = df_swbm[col_name].values / 1e6 # MNm
    
    ax.plot(pos, moments, color=COLORS['combined'], lw=2.5, label=col_name)
    ax.fill_between(pos, moments, color=COLORS['combined'], alpha=0.15)
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Position (m)', fontsize=10)
    ax.set_ylabel('SWBM (MN.m)', fontsize=10)
    ax.set_title('Still Water Bending Moment (SWBM) Profile', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_raw_wave_moments(time, raw_components, title):
    """
    Plots the raw wave bending moment time series (Vertical, Horizontal, Torsional).
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    for key, val in raw_components.items():
        color_key = key.lower()[:4]
        color = COLORS.get(color_key, '#7f7f7f')
        ax.plot(time, val / 1e6, color=color, lw=1.2, label=f'{key} WBM')
        
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Moment (MN.m)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_combined_dynamic_moments(time, combined_moments, title, is_beam_wave=False):
    """
    Plots the combined dynamic moments after applying Turkstra's combination rule.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    color = '#003366' if is_beam_wave else COLORS['vert']
    ax.plot(time, combined_moments / 1e6, color=color, lw=1.5, label='Combined WBM')
    ax.fill_between(time, combined_moments / 1e6, color=color, alpha=0.15)
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Combined Moment (MN.m)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_jpdf_overlay(s_mean, s_std, s_dist, l_mean, l_std, l_dist, W_modulus, title):
    """
    Plots the JPDF Overlay (Strength PDF vs Total Load Stress PDF).
    Converts Load Bending Moment parameters to Stress (MPa) using Section Modulus.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    # Calculate load stress statistics in MPa
    # Stress = (Moment in N.m / Modulus in m^3) * 1e-6
    mu_sigma_L = (l_mean / W_modulus) * 1e-6
    std_sigma_L = (l_std / W_modulus) * 1e-6
    
    # Define bounds for plotting
    lo = min(s_mean - 4.0 * s_std, mu_sigma_L - 4.0 * std_sigma_L)
    hi = max(s_mean + 4.0 * s_std, mu_sigma_L + 4.0 * std_sigma_L)
    x = np.linspace(lo, hi, 1000)
    
    # Load distribution (PDF)
    try:
        f_L = get_load_pdf(l_dist, mu_sigma_L, std_sigma_L)
        pdf_load = f_L(x)
    except Exception:
        # Fallback to normal
        pdf_load = norm.pdf(x, loc=mu_sigma_L, scale=std_sigma_L)
        
    # Strength distribution (PDF)
    # Get PDF from CDF derivative or build it
    if s_dist.lower() == 'normal':
        pdf_strength = norm.pdf(x, loc=s_mean, scale=s_std)
    elif s_dist.lower() == 'lognormal':
        cov = s_std / s_mean
        s_y = np.sqrt(np.log(1.0 + cov**2))
        mu_y = np.log(s_mean) - 0.5 * s_y**2
        pdf_strength = lognorm.pdf(x, s=s_y, scale=np.exp(mu_y))
    elif s_dist.lower() == 'weibull':
        from src.calculations import fit_weibull_from_mean_std
        k, scale = fit_weibull_from_mean_std(s_mean, s_std)
        pdf_strength = weibull_min.pdf(x, c=k, scale=scale)
    else:
        pdf_strength = norm.pdf(x, loc=s_mean, scale=s_std)
        
    ax.plot(x, pdf_load, color=COLORS['combined'], lw=2, label=f'Load Stress L ({l_dist})\nμ = {mu_sigma_L:.2f} MPa')
    ax.fill_between(x, pdf_load, color=COLORS['combined'], alpha=0.15)
    
    ax.plot(x, pdf_strength, color=COLORS['strength'], lw=2, label=f'Material Strength S ({s_dist})\nμ = {s_mean:.2f} MPa')
    ax.fill_between(x, pdf_strength, color=COLORS['strength'], alpha=0.15)
    
    ax.set_xlabel('Stress (MPa)', fontsize=10)
    ax.set_ylabel('Probability Density', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='upper right')
    
    plt.tight_layout()
    return fig

def plot_comparison_metric(results_df, metric='Beta_L3', title='Reliability Comparison'):
    """
    Plots a bar chart or line chart comparing a specific metric across different cases.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    # We can group by case description
    descriptions = results_df['Case Keterangan'].values
    values = results_df[metric].values
    
    bars = ax.barh(descriptions, values, color=COLORS['combined'], height=0.5, edgecolor='#002244')
    
    # Add values on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (width * 0.01), bar.get_y() + bar.get_height()/2.0, 
                f'{width:.4f}' if metric != 'Pf_L3' else f'{width:.2e}', 
                ha='left', va='center', fontsize=9, fontweight='bold')
        
    ax.set_xlabel(metric, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig
