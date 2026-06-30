# Ship Hull Girder Reliability Analysis (Streamlit App)

This is a Streamlit-based web application for analyzing the structural reliability of a ship's hull girder. It supports multi-component dynamic wave load combinations (using Turkstra's rule) and calculates reliability indices (Safety Index $\beta$ and Safety Factor FoS) alongside Level 3 convolution integrals for probability of failure ($P_f$).

## 🚀 How to Run Locally

1. **Activate your Python environment**:
   ```bash
   source .venv/bin/activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
- `app.py`: Entrypoint for the Streamlit UI.
- `src/`: Core Python modules containing the loader, statistics, plotting, and calculations.
- `data/examples/`: Bundled example CSV datasets (SWBM and wave time-series data).
- `walkthrough.md`: Detailed documentation of the mathematical and architectural components.
