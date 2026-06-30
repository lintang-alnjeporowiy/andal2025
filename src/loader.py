import pandas as pd
import numpy as np
import io

def detect_delimiter(file_buffer):
    """
    Detects whether the CSV file uses semicolons or commas.
    """
    # Read first line
    file_buffer.seek(0)
    first_line = file_buffer.readline()
    file_buffer.seek(0)
    if isinstance(first_line, bytes):
        first_line = first_line.decode('utf-8', errors='ignore')
    if ';' in first_line:
        return ';'
    return ','

def load_swbm_csv(file_buffer):
    """
    Loads SWBM CSV and extracts max absolute vertical SWBM.
    """
    sep = detect_delimiter(file_buffer)
    df = pd.read_csv(file_buffer, sep=sep)
    
    # Try to find vertical SWBM column
    col = None
    for c in df.columns:
        if 'vertical' in c.lower() and 'swbm' in c.lower():
            col = c
            break
    if col is None:
        # Fallback to any column with SWBM or Vertical
        for c in df.columns:
            if 'swbm' in c.lower() or 'vertical' in c.lower():
                col = c
                break
    if col is None:
        # Fallback to last column
        col = df.columns[-1]
        
    values = df[col].dropna().values
    max_val = np.max(np.abs(values))
    return df, col, max_val

def load_static_wbm_csv(file_buffer):
    """
    Loads Static Wave Bending Moment CSV and extracts max absolute value.
    """
    sep = detect_delimiter(file_buffer)
    df = pd.read_csv(file_buffer, sep=sep)
    
    # Try to find vertical or wave column
    col = None
    for c in df.columns:
        if 'vertical' in c.lower() or 'wave' in c.lower() or 'momen' in c.lower():
            col = c
            break
    if col is None:
        col = df.columns[-1]
        
    values = df[col].dropna().values
    max_val = np.max(np.abs(values))
    return df, col, max_val

def process_dynamic_wbm_csv(file_buffer, wave_direction):
    """
    Processes dynamic WBM CSV, identifies dominant/secondary/tertiary components based on wave direction,
    and applies correlation-weighted Turkstra load combination.
    """
    sep = detect_delimiter(file_buffer)
    df = pd.read_csv(file_buffer, sep=sep)
    
    # Find columns for Vertical, Horizontal, Torsional
    c_vert, c_horiz, c_tors = None, None, None
    for c in df.columns:
        c_low = c.lower()
        if 'vert' in c_low:
            c_vert = c
        elif 'horiz' in c_low:
            c_horiz = c
        elif 'tors' in c_low or 'tor' in c_low:
            c_tors = c
            
    # Fallback to index-based mapping if columns not found
    cols = [c for c in df.columns if c.lower() != 'time (s)' and 'time' not in c.lower()]
    if c_horiz is None and len(cols) >= 1:
        c_horiz = cols[0]
    if c_vert is None and len(cols) >= 2:
        c_vert = cols[1]
    if c_tors is None and len(cols) >= 3:
        c_tors = cols[2]
        
    # Map dominant/secondary/tertiary
    if wave_direction == 180:
        # Head wave: Vertical is dominant
        f1_col, f2_col, f3_col = c_vert, c_horiz, c_tors
    else:
        # Beam wave (90): Horizontal is dominant
        f1_col, f2_col, f3_col = c_horiz, c_vert, c_tors
        
    f1 = df[f1_col].values if f1_col else np.zeros(len(df))
    f2 = df[f2_col].values if f2_col else np.zeros(len(df))
    f3 = df[f3_col].values if f3_col else np.zeros(len(df))
    
    # Calculate correlations
    rho_12 = np.corrcoef(f1, f2)[0, 1] if len(f1) > 1 and f2_col else 0.0
    rho_13 = np.corrcoef(f1, f3)[0, 1] if len(f1) > 1 and f3_col else 0.0
    
    if np.isnan(rho_12): rho_12 = 0.0
    if np.isnan(rho_13): rho_13 = 0.0
    
    K2 = np.abs(rho_12)
    K3 = np.abs(rho_13)
    
    # Combine using absolute values (Turkstra Rule)
    wbm_dynamic = np.abs(f1) + K2 * np.abs(f2) + K3 * np.abs(f3)
    
    time_col = [c for c in df.columns if 'time' in c.lower()]
    time_vals = df[time_col[0]].values if time_col else np.arange(len(df))
    
    raw_components = {
        "Vertical": f1 if wave_direction == 180 else f2,
        "Horizontal": f2 if wave_direction == 180 else f1,
        "Torsional": f3
    }
    
    stats = {
        "mean_dynamic": np.mean(wbm_dynamic),
        "std_dynamic": np.std(wbm_dynamic, ddof=1),
        "K2": K2,
        "K3": K3,
        "rho_12": rho_12,
        "rho_13": rho_13
    }
    
    return df, time_vals, raw_components, wbm_dynamic, stats
