import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
from scipy.stats import norm, lognorm, weibull_min, rayleigh
import math

def fit_weibull_from_mean_std(mean, std):
    """
    Fits Weibull shape (k) and scale (lambda) parameters from mean and standard deviation.
    """
    cov = std / mean
    # Objective function to find k: Gamma(1+2/k) / Gamma(1+1/k)^2 - 1 - COV^2 = 0
    def obj(k):
        if k <= 0.01:
            return 1e6
        g1 = math.gamma(1.0 + 1.0 / k)
        g2 = math.gamma(1.0 + 2.0 / k)
        return (g2 / (g1 ** 2)) - 1.0 - (cov ** 2)

    # Solve for k
    try:
        sol = optimize.root_scalar(obj, bracket=[0.02, 100.0], method='brentq')
        k = sol.root
    except ValueError:
        # Fallback approximation
        k = 1.2 / (cov ** 1.05) if cov > 0 else 1.0
        
    g1 = math.gamma(1.0 + 1.0 / k)
    scale = mean / g1
    return k, scale

def get_strength_dist(dist_type, mean, std):
    """
    Returns the CDF function F_S(x) for the Strength distribution.
    """
    if dist_type.lower() == 'normal':
        return lambda x: norm.cdf(x, loc=mean, scale=std)
    elif dist_type.lower() == 'lognormal':
        cov = std / mean
        s_y = np.sqrt(np.log(1.0 + cov**2))
        mu_y = np.log(mean) - 0.5 * s_y**2
        # scipy.stats.lognorm: shape parameter is s_y, scale is exp(mu_y)
        return lambda x: lognorm.cdf(x, s=s_y, scale=np.exp(mu_y))
    elif dist_type.lower() == 'weibull':
        k, scale = fit_weibull_from_mean_std(mean, std)
        return lambda x: weibull_min.cdf(x, c=k, scale=scale)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

def get_load_pdf(dist_type, mean, std):
    """
    Returns the PDF function f_L(x) for the Load distribution.
    """
    if dist_type.lower() == 'normal':
        return lambda x: norm.pdf(x, loc=mean, scale=std)
    elif dist_type.lower() == 'rayleigh':
        # Rayleigh has mean = mode * sqrt(pi / 2). Let's fit mode:
        scale = mean / np.sqrt(np.pi / 2.0)
        return lambda x: rayleigh.pdf(x, scale=scale)
    elif dist_type.lower() == 'weibull':
        k, scale = fit_weibull_from_mean_std(mean, std)
        return lambda x: weibull_min.pdf(x, c=k, scale=scale)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

def calculate_reliability_level3(s_mean, s_std, s_dist, l_mean, l_std, l_dist):
    """
    Performs Level 3 reliability analysis using the convolution integral:
    Pf = integral_0^inf F_S(x) * f_L(x) dx
    """
    F_S = get_strength_dist(s_dist, s_mean, s_std)
    f_L = get_load_pdf(l_dist, l_mean, l_std)
    
    # Define the integrand
    def integrand(x):
        return F_S(x) * f_L(x)
    
    # We integrate over a reasonable range: from 0 to l_mean + 10 * l_std
    upper_bound = max(1.0, l_mean + 10.0 * l_std)
    pf, _ = integrate.quad(integrand, 0.0, upper_bound, limit=100)
    
    # Safety checks
    pf = max(1e-15, min(pf, 1.0))
    
    # Equivalent Beta: -norm.ppf(Pf)
    beta_eq = -norm.ppf(pf)
    return pf, beta_eq

def run_single_reliability(s_mean, s_std, s_dist, l_mean, l_std, l_dist):
    """
    Runs complete reliability analysis for a single set of strength & load distributions.
    Returns a dict with: FoS, Beta (analytical Normal), Pf (Level 3), and Beta_eq (Level 3).
    """
    # FoS
    fos = s_mean / l_mean if l_mean > 0 else np.nan
    
    # Analytical Beta (assuming both Normal)
    beta_normal = (s_mean - l_mean) / np.sqrt(s_std**2 + l_std**2)
    
    # Level 3 convolution
    pf_l3, beta_l3 = calculate_reliability_level3(s_mean, s_std, s_dist, l_mean, l_std, l_dist)
    
    return {
        "FoS": fos,
        "Beta_Normal": beta_normal,
        "Pf_L3": pf_l3,
        "Beta_L3": beta_l3
    }
