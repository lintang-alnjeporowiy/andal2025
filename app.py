import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import math
from src.loader import load_swbm_csv, load_static_wbm_csv, process_dynamic_wbm_csv
from src.calculations import run_single_reliability, fit_weibull_from_mean_std
from src.plotting import (
    plot_swbm_static,
    plot_raw_wave_moments,
    plot_combined_dynamic_moments,
    plot_jpdf_overlay,
    plot_comparison_metric
)

# App Title & Layout
st.set_page_color = None
st.set_page_config(page_title="Ship Hull Reliability Analysis", layout="wide")

# CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-title {
        color: #003366;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #eef;
    }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-title">Ship Structural Reliability Calculator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Perform structural reliability calculations</div>', unsafe_allow_html=True)

# ----------------- SESSION STATE INITIALIZATION -----------------
if 'materials' not in st.session_state:
    st.session_state['materials'] = []
if 'swbm_cases' not in st.session_state:
    st.session_state['swbm_cases'] = []
if 'static_wbm_cases' not in st.session_state:
    st.session_state['static_wbm_cases'] = []
if 'wave_cases' not in st.session_state:
    st.session_state['wave_cases'] = []
if 'section_moduli' not in st.session_state:
    st.session_state['section_moduli'] = [3.2292]
if 'swbm_ship' not in st.session_state:
    st.session_state['swbm_ship'] = 1.2e8 # 120,000 kNm in N.m
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'execution_details' not in st.session_state:
    st.session_state['execution_details'] = {}

# Example files path helper
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'examples')

def load_defaults():
    """Populates session state with default data mimicking the main notebook analysis."""
    st.session_state['section_moduli'] = [3.2292]
    st.session_state['swbm_ship'] = 1.2e8
    
    # Default materials
    st.session_state['materials'] = [
        {"name": "A36 Steel (Yield)", "mean": 250.0, "std": 25.0, "dist": "Normal"},
        {"name": "A36 Steel (Ultimate)", "mean": 414.93, "std": 57.65, "dist": "Normal"}
    ]
    
    # Default SWBM cases
    st.session_state['swbm_cases'] = [
        {"name": "Default SWBM", "val": 1.5464e8, "unit": "N.m"}
    ]
    
    # Default Static WBM cases
    st.session_state['static_wbm_cases'] = [
        {"name": "Zero Static WBM", "val": 0.0, "unit": "N.m"}
    ]
    
    # Default Wave Cases (6 configurations using example CSVs)
    st.session_state['wave_cases'] = [
        # Note: mapping is adjusted for physical wave height consistency (Hs-1.73 CSV has Hs = 2.58 m data)
        {"file_type": "Example File", "filename": "wbm/Hs-1.73_D-180.csv", "Hs": 2.58, "D": 180, "dist": "Normal", "name": "Hs=2.58m, D=180°"},
        {"file_type": "Example File", "filename": "wbm/Hs-1.73_D-90.csv", "Hs": 2.58, "D": 90, "dist": "Normal", "name": "Hs=2.58m, D=90°"},
        {"file_type": "Example File", "filename": "wbm/Hs-2.16_D-180.csv", "Hs": 2.16, "D": 180, "dist": "Normal", "name": "Hs=2.16m, D=180°"},
        {"file_type": "Example File", "filename": "wbm/Hs-2.16_D-90.csv", "Hs": 2.16, "D": 90, "dist": "Normal", "name": "Hs=2.16m, D=90°"},
        {"file_type": "Example File", "filename": "wbm/Hs-2.58_D-180.csv", "Hs": 1.73, "D": 180, "dist": "Normal", "name": "Hs=1.73m, D=180°"},
        {"file_type": "Example File", "filename": "wbm/Hs-2.58_D-90.csv", "Hs": 1.73, "D": 90, "dist": "Normal", "name": "Hs=1.73m, D=90°"},
    ]

# ----------------- SIDEBAR CONFIGURATIONS -----------------
with st.sidebar:
    st.header("Global Parameters")
    st.session_state['swbm_ship'] = st.number_input(
        "Ship SWBM Constant (N.m)",
        min_value=0.0,
        value=st.session_state['swbm_ship'],
        step=1.0e6,
        format="%.4e",
        help="Nilai momen still water bending moment bawaan kapal (SWBM kapal)"
    )
    
    st.markdown("---")
    if st.button("Load Default Notebook Data", use_container_width=True):
        load_defaults()
        st.success("Default configuration loaded!")
        st.rerun()
        
    st.markdown("### Quick Examples Info")
    st.info("Default data contains 1 Section Modulus value, 2 materials, 1 SWBM, 1 Static WBM, and 6 wave cases, totaling 12 calculation runs.")

# ----------------- MAIN INTERFACE TABS -----------------
tab1, tab2, tab3 = st.tabs(["Setup & Configurations", "Analysis Results", "Visualizations"])

with tab1:
    col_l, col_r = st.columns(2)
    
    with col_l:
        # 1. MATERIALS MANAGEMENT
        st.subheader("1. Material Strengths")
        with st.container(border=True):
            mat_name = st.text_input("Material Name", "A36 Steel")
            mat_dist = st.selectbox("Probability Distribution", ["Normal", "Lognormal", "Weibull"], key="mat_dist_select", help="Pilih jenis distribusi probabilitas untuk kekuatan material.")
            mat_input_method = st.radio("Input Method", ["Manual Parameters", "Upload CSV", "Use A36 Example Data"], key="mat_method")
            
            mean_mat, std_mat = 0.0, 0.0
            uploaded_mat_df = None
            
            if mat_input_method == "Manual Parameters":
                if mat_dist == "Normal":
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        mean_mat = st.number_input(
                            "Mean Strength (μ) (MPa)", 
                            min_value=1.0, 
                            value=250.0,
                            help="Nilai rata-rata aritmatika kekuatan material."
                        )
                    with col_m2:
                        param_type = st.selectbox("Dispersion parameter", ["Standard Deviation", "COV (Coeff of Var)"])
                        if param_type == "Standard Deviation":
                            std_mat = st.number_input(
                                "Std Dev (σ) (MPa)", 
                                min_value=0.1, 
                                value=25.0,
                                help="Standar deviasi aritmatika kekuatan material."
                            )
                        else:
                            cov_val = st.number_input("COV (0 to 1)", min_value=0.01, max_value=1.0, value=0.1)
                            std_mat = mean_mat * cov_val
                            
                elif mat_dist == "Lognormal":
                    log_input_mode = st.radio(
                        "Lognormal Parameters Input Mode",
                        ["Arithmetic (Mean & Std Dev)", "Log-scale (μ_ln & σ_ln)"],
                        help="Pilih metode input untuk distribusi Lognormal. Baik parameter aritmatika maupun log-skala secara otomatis dikonversi secara presisi."
                    )
                    if log_input_mode == "Arithmetic (Mean & Std Dev)":
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            mean_mat = st.number_input(
                                "Mean Strength (μ) (MPa)", 
                                min_value=1.0, 
                                value=250.0,
                                help="Nilai rata-rata aritmatika kekuatan material."
                            )
                        with col_m2:
                            std_mat = st.number_input(
                                "Std Dev (σ) (MPa)", 
                                min_value=0.1, 
                                value=25.0,
                                help="Standar deviasi aritmatika kekuatan material."
                            )
                    else:
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            mu_ln = st.number_input(
                                "Log-scale Mean (μ_ln)", 
                                value=5.5, 
                                step=0.1, 
                                format="%.4f",
                                help="Parameter rata-rata dalam skala logaritma natural (μ_ln = E[ln(X)])."
                            )
                        with col_m2:
                            sigma_ln = st.number_input(
                                "Log-scale Std Dev (σ_ln)", 
                                min_value=0.01, 
                                value=0.1, 
                                step=0.01, 
                                format="%.4f",
                                help="Parameter deviasi standar dalam skala logaritma natural (σ_ln = Std[ln(X)])."
                            )
                        mean_mat = np.exp(mu_ln + 0.5 * sigma_ln**2)
                        std_mat = np.sqrt((np.exp(sigma_ln**2) - 1.0) * np.exp(2.0 * mu_ln + sigma_ln**2))
                        st.info(f"Equivalent Arithmetic Stats: Mean = **{mean_mat:.2f} MPa**, Std Dev = **{std_mat:.2f} MPa**")
                        
                elif mat_dist == "Weibull":
                    weib_input_mode = st.radio(
                        "Weibull Parameters Input Mode",
                        ["Arithmetic (Mean & Std Dev)", "Weibull Parameters (Shape k & Scale λ)"],
                        help="Pilih metode input untuk distribusi Weibull. Parameter bentuk (shape) dan skala (scale) akan dikonversi dengan rumus Gamma."
                    )
                    if weib_input_mode == "Arithmetic (Mean & Std Dev)":
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            mean_mat = st.number_input(
                                "Mean Strength (μ) (MPa)", 
                                min_value=1.0, 
                                value=250.0,
                                help="Nilai rata-rata aritmatika kekuatan material."
                            )
                        with col_m2:
                            std_mat = st.number_input(
                                "Std Dev (σ) (MPa)", 
                                min_value=0.1, 
                                value=25.0,
                                help="Standar deviasi aritmatika kekuatan material."
                            )
                    else:
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            shape_k = st.number_input(
                                "Shape Parameter (k)", 
                                min_value=0.1, 
                                value=10.0, 
                                step=0.5, 
                                format="%.2f",
                                help="Parameter bentuk (shape parameter c/k pada scipy/Wikipedia) yang menentukan kelengkungan distribusi."
                            )
                        with col_m2:
                            scale_lam = st.number_input(
                                "Scale Parameter (λ) (MPa)", 
                                min_value=1.0, 
                                value=260.0, 
                                step=10.0, 
                                format="%.2f",
                                help="Parameter skala (scale parameter λ) dalam MPa."
                            )
                        mean_mat = scale_lam * math.gamma(1.0 + 1.0 / shape_k)
                        std_mat = np.sqrt(scale_lam**2 * (math.gamma(1.0 + 2.0 / shape_k) - math.gamma(1.0 + 1.0 / shape_k)**2))
                        st.info(f"Equivalent Arithmetic Stats: Mean = **{mean_mat:.2f} MPa**, Std Dev = **{std_mat:.2f} MPa**")
            
            elif mat_input_method == "Upload CSV":
                mat_csv = st.file_uploader(
                    "Upload Strength CSV (in MPa)", 
                    type=["csv"], 
                    key="mat_csv_uploader",
                    help="Format CSV: Harus memiliki kolom berisi nilai kekuatan material (dalam MPa). Baris pertama harus berisi nama header kolom."
                )
                if mat_csv:
                    df = pd.read_csv(mat_csv)
                    st.dataframe(df.head(3), width="stretch")
                    col_col = st.selectbox("Select Strength Column", df.columns)
                    values = df[col_col].dropna().values
                    mean_mat = np.mean(values)
                    std_mat = np.std(values, ddof=1)
                    st.write(f"Parsed Stats: Mean = **{mean_mat:.2f} MPa**, Std Dev = **{std_mat:.2f} MPa**")
            
            else:
                st.info("Uses ultimate strength data from A36 Steel (Mean = 414.93 MPa, Std = 57.65 MPa)")
                mean_mat = 414.93
                std_mat = 57.65
                
            if st.button("Add Material", use_container_width=True):
                st.session_state['materials'].append({
                    "name": mat_name,
                    "mean": mean_mat,
                    "std": std_mat,
                    "dist": mat_dist
                })
                st.success(f"Added {mat_name}!")
                
        # Display Current Materials
        if st.session_state['materials']:
            st.markdown("**Current Materials:**")
            mat_df = pd.DataFrame(st.session_state['materials'])
            st.dataframe(mat_df, width="stretch")
            if st.button("Clear Materials"):
                st.session_state['materials'] = []
                st.rerun()

        # 2. SWBM & STATIC WBM
        st.subheader("2. Still Water & Static Bending Moments")
        with st.container(border=True):
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.write("**Add SWBM Case**")
                swbm_name = st.text_input("SWBM Name", "Load Case 1")
                swbm_val_input = st.number_input("SWBM Value", min_value=0.0, value=154.64, format="%.4f")
                swbm_unit = st.selectbox("SWBM Unit", ["MN.m", "N.m"])
                swbm_val = swbm_val_input * 1e6 if swbm_unit == "MN.m" else swbm_val_input
                
                if st.button("Add SWBM", use_container_width=True):
                    st.session_state['swbm_cases'].append({
                        "name": swbm_name,
                        "val": swbm_val,
                        "unit": "N.m"
                    })
                    st.success(f"Added SWBM: {swbm_name}")
            
            with col_b2:
                st.write("**Add Static WBM Case**")
                static_name = st.text_input("Static WBM Name", "Zero Static")
                static_method = st.radio("Static Input Source", ["Manual Value", "Upload CSV"])
                
                static_val = 0.0
                if static_method == "Manual Value":
                    static_val_input = st.number_input("Static WBM Value", min_value=0.0, value=0.0, format="%.4f")
                    static_unit = st.selectbox("Static Unit", ["MN.m", "N.m"])
                    static_val = static_val_input * 1e6 if static_unit == "MN.m" else static_val_input
                else:
                    static_csv = st.file_uploader(
                        "Upload Static WBM CSV", 
                        type=["csv"], 
                        key="static_csv_uploader",
                        help="Format CSV: Harus memiliki kolom berisi nilai momen lentur statis (dalam N.m atau MN.m). Kolom dicari otomatis dengan kata kunci 'vertical', 'wave', atau 'momen'."
                    )
                    if static_csv:
                        _, _, max_static = load_static_wbm_csv(static_csv)
                        static_val = max_static
                        st.write(f"Parsed Max Absolute Static WBM: **{static_val/1e6:.2f} MN.m**")
                
                if st.button("Add Static WBM", use_container_width=True):
                    st.session_state['static_wbm_cases'].append({
                        "name": static_name,
                        "val": static_val,
                        "unit": "N.m"
                    })
                    st.success(f"Added Static WBM: {static_name}")
                    
        # Display Moments
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if st.session_state['swbm_cases']:
                st.write("SWBM Cases:")
                st.dataframe(pd.DataFrame(st.session_state['swbm_cases']), width="stretch")
                if st.button("Clear SWBM"):
                    st.session_state['swbm_cases'] = []
                    st.rerun()
        with col_d2:
            if st.session_state['static_wbm_cases']:
                st.write("Static WBM Cases:")
                st.dataframe(pd.DataFrame(st.session_state['static_wbm_cases']), width="stretch")
                if st.button("Clear Static WBM"):
                    st.session_state['static_wbm_cases'] = []
                    st.rerun()

    with col_r:
        # 3. DYNAMIC WAVE BENDING MOMENTS
        st.subheader("3. Dynamic Wave Bending Moments")
        with st.container(border=True):
            wave_name = st.text_input("Wave Case Keterangan", "Head Wave Hs=2.58m")
            wave_dist = st.selectbox("Fit/Model Load Distribution", ["Normal", "Rayleigh", "Weibull"], key="wave_dist_select", help="Pilih jenis distribusi probabilitas untuk memodelkan beban momen dinamis gelombang.")
            wave_source = st.radio("Dynamic Load Source", ["Manual Parameters", "Upload CSV", "Use Example File"], key="wave_source")
            
            wave_hs = st.number_input("Wave Height Hs (m)", min_value=0.1, value=2.58)
            wave_dir = st.selectbox("Wave Direction D (deg)", [180, 90], help="180: Head Wave, 90: Beam Wave")
            
            # Setup inputs based on source
            wave_params = {}
            if wave_source == "Manual Parameters":
                if wave_dist == "Normal":
                    col_w1, col_w2 = st.columns(2)
                    with col_w1:
                        mu_d = st.number_input(
                            "Mean Dynamic Moment (MN.m)", 
                            min_value=0.0, 
                            value=15.0,
                            help="Nilai rata-rata momen dinamis gelombang."
                        )
                    with col_w2:
                        std_d = st.number_input(
                            "Std Dev Dynamic Moment (MN.m)", 
                            min_value=0.1, 
                            value=5.0,
                            help="Standar deviasi momen dinamis gelombang."
                        )
                    wave_params = {"file_type": "Manual", "mean": mu_d * 1e6, "std": std_d * 1e6}
                    
                elif wave_dist == "Rayleigh":
                    ray_input_mode = st.radio(
                        "Rayleigh Parameters Input Mode",
                        ["Arithmetic Mean (MN.m)", "Rayleigh Scale Parameter (σ) (MN.m)"],
                        help="Pilih metode input untuk distribusi Rayleigh. Parameter skala akan dikonversi dengan rumus Rayleigh."
                    )
                    if ray_input_mode == "Arithmetic Mean (MN.m)":
                        mu_d = st.number_input(
                            "Mean Dynamic Moment (MN.m)", 
                            min_value=0.01, 
                            value=15.0,
                            help="Nilai rata-rata momen dinamis gelombang."
                        )
                        scale_val = mu_d / np.sqrt(np.pi / 2.0)
                        std_d = scale_val * np.sqrt(2.0 - np.pi / 2.0)
                    else:
                        scale_val = st.number_input(
                            "Rayleigh Scale Parameter (σ) (MN.m)", 
                            min_value=0.01, 
                            value=12.0, 
                            step=1.0,
                            help="Parameter skala (σ) untuk distribusi Rayleigh. Hubungan dengan tinggi gelombang individual jangka pendek."
                        )
                        mu_d = scale_val * np.sqrt(np.pi / 2.0)
                        std_d = scale_val * np.sqrt(2.0 - np.pi / 2.0)
                        
                    st.info(f"Equivalent Arithmetic Stats: Mean = **{mu_d:.2f} MN.m**, Std Dev = **{std_d:.2f} MN.m**")
                    wave_params = {"file_type": "Manual", "mean": mu_d * 1e6, "std": std_d * 1e6}
                    
                elif wave_dist == "Weibull":
                    weib_input_mode = st.radio(
                        "Weibull Parameters Input Mode (Wave)",
                        ["Arithmetic (Mean & Std Dev) (MN.m)", "Weibull Parameters (Shape k & Scale λ) (MN.m)"],
                        help="Pilih metode input untuk distribusi Weibull."
                    )
                    if weib_input_mode == "Arithmetic (Mean & Std Dev) (MN.m)":
                        col_w1, col_w2 = st.columns(2)
                        with col_w1:
                            mu_d = st.number_input(
                                "Mean Dynamic Moment (MN.m)", 
                                min_value=0.0, 
                                value=15.0,
                                help="Nilai rata-rata momen dinamis gelombang."
                            )
                        with col_w2:
                            std_d = st.number_input(
                                "Std Dev Dynamic Moment (MN.m)", 
                                min_value=0.1, 
                                value=5.0,
                                help="Standar deviasi momen dinamis gelombang."
                            )
                    else:
                        col_w1, col_w2 = st.columns(2)
                        with col_w1:
                            shape_k = st.number_input(
                                "Shape Parameter (k)", 
                                min_value=0.1, 
                                value=1.8, 
                                step=0.1, 
                                format="%.2f",
                                help="Parameter bentuk Weibull (c/k pada scipy/Wikipedia)."
                            )
                        with col_w2:
                            scale_lam = st.number_input(
                                "Scale Parameter (λ) (MN.m)", 
                                min_value=0.1, 
                                value=16.5, 
                                step=1.0, 
                                format="%.2f",
                                help="Parameter skala Weibull (λ) dalam MN.m."
                            )
                        mu_d = scale_lam * math.gamma(1.0 + 1.0 / shape_k)
                        std_d = np.sqrt(scale_lam**2 * (math.gamma(1.0 + 2.0 / shape_k) - math.gamma(1.0 + 1.0 / shape_k)**2))
                        
                    st.info(f"Equivalent Arithmetic Stats: Mean = **{mu_d:.2f} MN.m**, Std Dev = **{std_d:.2f} MN.m**")
                    wave_params = {"file_type": "Manual", "mean": mu_d * 1e6, "std": std_d * 1e6}
                
            elif wave_source == "Upload CSV":
                wave_csv = st.file_uploader(
                    "Upload Time-Series CSV (N.m)", 
                    type=["csv"], 
                    key="wave_csv_uploader",
                    help="Format CSV: Runtun waktu (time-series) momen dinamis kapal. Harus memiliki kolom waktu (misalnya 'Time (s)') dan kolom momen lentur dinamis: 'Vertical', 'Horizontal', dan 'Torsional' (atau minimal salah satunya untuk perhitungan Turkstra berdasarkan arah gelombang)."
                )
                wave_params = {"file_type": "Upload", "file": wave_csv}
                
            else: # Example File
                examples_list = [f for f in os.listdir(os.path.join(EXAMPLES_DIR, 'wbm')) if f.endswith('.csv')]
                selected_ex = st.selectbox("Select Example CSV", sorted(examples_list))
                wave_params = {"file_type": "Example File", "filename": os.path.join('wbm', selected_ex)}
            
            if st.button("Add Wave Case", use_container_width=True):
                st.session_state['wave_cases'].append({
                    "name": wave_name,
                    "Hs": wave_hs,
                    "D": wave_dir,
                    "dist": wave_dist,
                    **wave_params
                })
                st.success(f"Added Wave Case: {wave_name}!")
                
        # Display Current Wave Cases
        if st.session_state['wave_cases']:
            st.markdown("**Current Wave Cases:**")
            wave_display = []
            for w in st.session_state['wave_cases']:
                wave_display.append({
                    "Name": w["name"],
                    "Hs (m)": w["Hs"],
                    "D (deg)": w["D"],
                    "Dist Type": w["dist"],
                    "Source": w["file_type"]
                })
            st.dataframe(pd.DataFrame(wave_display), width="stretch")
            if st.button("Clear Wave Cases"):
                st.session_state['wave_cases'] = []
                st.rerun()

        # 4. SECTION MODULI (W)
        st.subheader("4. Section Moduli (W)")
        with st.container(border=True):
            col_w1, col_w2 = st.columns([3, 1])
            with col_w1:
                new_w = st.number_input("Add Section Modulus W (m³)", min_value=0.01, value=3.2292, step=0.1, key="input_modulus_w")
            with col_w2:
                st.write("") # spacing
                st.write("") # spacing
                add_w_btn = st.button("Add W", use_container_width=True, key="add_w_btn_action")
                
            if add_w_btn:
                if new_w not in st.session_state['section_moduli']:
                    st.session_state['section_moduli'].append(new_w)
                    st.success(f"Added W: {new_w} m³")
                    st.rerun()
                else:
                    st.warning(f"W: {new_w} m³ already exists.")
                    
        # Display Current Section Moduli
        if st.session_state['section_moduli']:
            st.markdown("**Current Section Moduli (m³):**")
            w_df = pd.DataFrame({"W (m³)": st.session_state['section_moduli']})
            st.dataframe(w_df, width="stretch")
            if st.button("Clear Section Moduli"):
                st.session_state['section_moduli'] = []
                st.rerun()

# ----------------- VARIATION CHECKLISTS & EXECUTE -----------------
st.markdown("---")
st.subheader("5. Run Simulation & Reliability Analysis")

if not st.session_state['materials'] or not st.session_state['swbm_cases'] or not st.session_state['static_wbm_cases'] or not st.session_state['wave_cases'] or not st.session_state['section_moduli']:
    st.warning("Please configure at least one item in each category (Materials, Section Moduli, SWBM, Static WBM, Wave Cases) to run the simulation.")
else:
    # Build complete grid of runs
    runs = []
    run_names = []
    for m in st.session_state['materials']:
        for w in st.session_state['section_moduli']:
            for sw in st.session_state['swbm_cases']:
                for st_w in st.session_state['static_wbm_cases']:
                    for wa in st.session_state['wave_cases']:
                        run_name = f"W: {w} m³ | Mat: {m['name']} | SWBM: {sw['name']} | Static: {st_w['name']} | Wave: {wa['name']}"
                        runs.append((m, w, sw, st_w, wa))
                        run_names.append(run_name)
                    
    st.markdown("**Select Configurations to Analyze:**")
    selected_runs_idx = []
    
    # Checkbox to toggle select all / deselect all
    select_all = st.checkbox("Select All / Deselect All Variations", value=True, key="select_all_chk")
    
    # Display individual checkboxes for each run inside an expander
    with st.expander("Choose specific variations to run", expanded=True):
        col_v1, col_v2 = st.columns(2)
        for idx, run_name in enumerate(run_names):
            col = col_v1 if idx % 2 == 0 else col_v2
            with col:
                is_selected = st.checkbox(run_name, value=select_all, key=f"run_chk_{idx}")
                if is_selected:
                    selected_runs_idx.append(idx)
                    
    st.write(f"Total variations selected to execute: **{len(selected_runs_idx)}**")
    
    # Run analysis button
    if st.button("Start Reliability Analysis", type="primary", use_container_width=True):
        with st.spinner("Executing structural reliability convolution pipeline..."):
            results_list = []
            execution_details = {}
            
            for idx in selected_runs_idx:
                m, w, sw, st_w, wa = runs[idx]
                
                # Retrieve/Process Wave Data
                mu_d, std_d = 0.0, 0.0
                time_series = None
                raw_components = None
                combined_moments = None
                
                if wa["file_type"] == "Manual":
                    mu_d = wa["mean"]
                    std_d = wa["std"]
                else:
                    # Load CSV
                    try:
                        if wa["file_type"] == "Example File":
                            file_path = os.path.join(EXAMPLES_DIR, wa["filename"])
                            with open(file_path, 'rb') as f:
                                file_buffer = io.BytesIO(f.read())
                        else:
                            # Uploaded file
                            file_buffer = io.BytesIO(wa["file"].getvalue())
                            
                        # Process WBM components using Turkstra's Rule
                        _, time_vals, raw_comp, combined_mom, w_stats = process_dynamic_wbm_csv(file_buffer, wa["D"])
                        
                        mu_d = w_stats["mean_dynamic"]
                        # Apply std dev correction
                        std_d = w_stats["std_dynamic"]
                        
                        time_series = time_vals
                        raw_components = raw_comp
                        combined_moments = combined_mom
                        
                        # Store timeseries for plotting later
                        execution_details[run_names[idx]] = {
                            "time": time_vals,
                            "raw_components": raw_comp,
                            "combined": combined_mom,
                            "stats": w_stats
                        }
                    except Exception as e:
                        st.error(f"Error loading wave case {wa['name']}: {str(e)}")
                        continue
                
                # Combine Loads: WBM_total = dynamic + SWBM + SWBM_ship + Static WBM
                # Since SWBM and Static WBM are treated as extreme values/maximums, we sum them
                # Total Moment = Mean dynamic + SWBM + Static WBM + Ship SWBM
                mu_moment = mu_d + sw["val"] + st_w["val"] + st.session_state['swbm_ship']
                
                # User correction checkbox for std dev: std_L = std + mu/8
                std_moment = std_d + mu_moment / 8.0
                
                # Run reliability calculations
                W = w
                mu_stress = (mu_moment / W) * 1e-6
                std_stress = (std_moment / W) * 1e-6
                
                rel_results = run_single_reliability(
                    s_mean=m["mean"],
                    s_std=m["std"],
                    s_dist=m["dist"],
                    l_mean=mu_stress,
                    l_std=std_stress,
                    l_dist=wa["dist"]
                )
                
                # Format failure probability friendly representation
                pf = rel_results["Pf_L3"]
                if pf == 0.0:
                    pf_friendly = "0"
                elif pf < 0.01:
                    exp = int(np.floor(np.log10(pf)))
                    base = pf / (10 ** exp)
                    pf_friendly = f"{base:.1f} x 10^{exp}"
                else:
                    pf_friendly = f"{pf:.4f}"
                    
                results_list.append({
                    "Case Description": run_names[idx],
                    "Case Keterangan": wa["name"],
                    "W (m³)": W,
                    "Material": m["name"],
                    "Mean Strength (MPa)": m["mean"],
                    "FoS": rel_results["FoS"],
                    "Analytical Beta": rel_results["Beta_Normal"],
                    "Level 3 Beta (Eq)": rel_results["Beta_L3"],
                    "Peluang Kegagalan": pf_friendly,
                    "Pf_Numeric": pf,
                    # Extra fields for plotting
                    "mu_moment": mu_moment,
                    "std_moment": std_moment,
                    "s_dist": m["dist"],
                    "s_std": m["std"],
                    "l_dist": wa["dist"]
                })
                
            st.session_state['results'] = pd.DataFrame(results_list)
            st.session_state['execution_details'] = execution_details
            st.success("Calculations complete! Review the results in the Tabs.")

# ----------------- RESULTS TABS DISPLAY -----------------
with tab2:
    st.subheader("Reliability Summary Table")
    if st.session_state['results'] is not None:
        df_res = st.session_state['results']
        
        # Display table with select columns
        display_cols = ["Case Description", "W (m³)", "Material", "Mean Strength (MPa)", "FoS", "Analytical Beta", "Level 3 Beta (Eq)", "Peluang Kegagalan"]
        st.dataframe(df_res[display_cols], width="stretch")
        
        # Download button
        csv_buffer = io.StringIO()
        df_res[display_cols].to_csv(csv_buffer, index=False)
        st.download_button(
            "Download CSV Report",
            data=csv_buffer.getvalue(),
            file_name="ship_reliability_report.csv",
            mime="text/csv"
        )
        
        # Show key metrics in columns for the worst case (minimum Beta)
        st.markdown("### Worst Performing Configuration")
        idx_min = df_res["Level 3 Beta (Eq)"].idxmin()
        worst_row = df_res.iloc[idx_min]
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Min Safety Index (Beta)", f"{worst_row['Level 3 Beta (Eq)']:.4f}", help="Indeks keandalan minimum")
        col_m2.metric("Min Factor of Safety", f"{worst_row['FoS']:.3f}")
        col_m3.metric("Max Probability of Failure", worst_row['Peluang Kegagalan'])
        st.info(f"Worst case: **{worst_row['Case Description']}**")
    else:
        st.info("Run the reliability analysis on the 'Setup' tab to view results.")

with tab3:
    st.subheader("Visualization Center")
    
    if st.session_state['results'] is not None:
        # Toggle switch to prevent Matplotlib lag during input editing
        enable_plots = st.toggle("Enable Plotting & Charts", value=False, help="Aktifkan ini untuk melihat grafik hasil analisis. Nonaktifkan ketika sedang memasukkan atau mengedit data input untuk performa maksimal (mencegah lag).")
        
        if not enable_plots:
            st.info("Centang/aktifkan tombol 'Enable Plotting & Charts' di atas untuk merender visualisasi grafik hasil analisis.")
        else:
            df_res = st.session_state['results']
            details = st.session_state['execution_details']
            
            # Select case to plot
            selected_plot_case = st.selectbox("Select Case to Visualize in detail", df_res["Case Description"].values)
            row_plot = df_res[df_res["Case Description"] == selected_plot_case].iloc[0]
            
            # Check if the selected plot case has timeseries details
            has_details = (selected_plot_case in details)
            
            # Visualisation options checklists
            st.markdown("##### Select Plots to Render:")
            col_ch1, col_ch2, col_ch3 = st.columns(3)
            
            if has_details:
                show_raw = col_ch1.checkbox("Raw Time-Series Wave Moments", value=True)
                show_comb = col_ch2.checkbox("Combined Turkstra Moment", value=True)
            else:
                show_raw = col_ch1.checkbox("Raw Time-Series Wave Moments (Not Available)", value=False, disabled=True, help="Hanya tersedia jika input menggunakan file CSV waktu dinamis.")
                show_comb = col_ch2.checkbox("Combined Turkstra Moment (Not Available)", value=False, disabled=True, help="Hanya tersedia jika input menggunakan file CSV waktu dinamis.")
                
            show_jpdf = col_ch3.checkbox("JPDF Overlay (Strength vs Stress)", value=True)
            
            # 1. Raw Time Series
            if show_raw and has_details:
                det = details[selected_plot_case]
                st.markdown(f"#### 1. Raw Wave Bending Moments - {selected_plot_case}")
                fig_raw = plot_raw_wave_moments(det["time"], det["raw_components"], f"Raw Wave Bending Moments: {row_plot['Case Keterangan']}")
                st.pyplot(fig_raw)
                
            # 2. Combined Moments
            if show_comb and has_details:
                det = details[selected_plot_case]
                st.markdown(f"#### 2. Combined Turkstra Moment - {selected_plot_case}")
                is_beam = "D=90°" in selected_plot_case or "D = 90" in selected_plot_case
                fig_comb = plot_combined_dynamic_moments(det["time"], det["combined"], f"Combined Turkstra Wave Moment: {row_plot['Case Keterangan']}", is_beam)
                st.pyplot(fig_comb)
                
            # 3. JPDF Overlay
            if show_jpdf:
                st.markdown(f"#### 3. JPDF Overlay (MPa) - {selected_plot_case}")
                fig_jpdf = plot_jpdf_overlay(
                    s_mean=row_plot["Mean Strength (MPa)"],
                    s_std=row_plot["s_std"],
                    s_dist=row_plot["s_dist"],
                    l_mean=row_plot["mu_moment"],
                    l_std=row_plot["std_moment"],
                    l_dist=row_plot["l_dist"],
                    W_modulus=row_plot["W (m³)"],
                    title=f"JPDF Overlay: {row_plot['Case Keterangan']}"
                )
                st.pyplot(fig_jpdf)
                
            # Global comparison chart
            st.markdown("---")
            st.markdown("#### Global Comparison (Level 3 Safety Index $\\beta$)")
            fig_comp = plot_comparison_metric(df_res, 'Level 3 Beta (Eq)', 'Safety Index (Beta) Comparison across All Cases')
            st.pyplot(fig_comp)
            
    else:
        st.info("Run the reliability analysis on the 'Setup' tab to unlock visualizations.")
