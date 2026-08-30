  # ⚡ Space Radiation Mission Control & Forecaster
### Physics-Informed Multi-Horizon Machine Learning for Relativistic Electron Flux at GEO

An operational space weather forecasting platform designed to predict relativistic outer radiation belt electron flux ($>2\text{ MeV}$) at geosynchronous orbit (GEO). The system couples upstream interplanetary solar wind conditions with geomagnetic storm indices to issue actionable hazard forecasts at **+1h, +6h, and +24h** lead times.

---

## 🌌 Scientific Problem & Operational Context

High-energy electrons ($>2\text{ MeV}$) trapped in Earth's outer Van Allen radiation belt pose critical operational risks to commercial and defense satellites:
* **Deep Dielectric Charging:** Relativistic electrons penetrate spacecraft shielding, accumulating charge inside circuit boards and insulators until electrostatic discharge triggers catastrophic component failure.
* **Non-Linear Acceleration:** Flux levels can surge by 4 to 6 orders of magnitude over hours due to solar wind-magnetosphere interactions (magnetic reconnection, substorm injections, and wave-particle interactions).

This system provides early warnings before relativistic electrons reach the hazardous operational threshold ($\ge 10^3\text{ cm}^{-2}\text{s}^{-1}\text{sr}^{-1}$).

---

## 🛰️ Multi-Mission Dataset Architecture

The platform harmonizes 4 years of continuous, multi-satellite observations onto a unified **5-minute temporal grid**:

| Instrument / Telemetry | Orbit / Location | Data Type | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| **NASA WIND (SWE & MFI)** | Sun-Earth L1 Lagrange ($\approx 1.5\text{M km}$ upstream) | $B_x, B_y, B_z, B_t$, Solar wind speed ($v_{sw}$), Density ($n_p$), Temp | Primary physical drivers |
| **NASA OMNI2** | Near-Earth Magnetosphere | $D_{st}$ (Ring current index), $K_p$ (Planetary geomagnetic index) | Global storm activity |
| **NOAA GOES-15 (EPEAD)** | GEO ($135^\circ\text{ W}$) | In-situ Relativistic Electron Flux ($>2\text{ MeV}$) | Training target & baseline |
| **ISRO GSAT-19 (GRASP)** | GEO ($48^\circ\text{ E}$) | High-energy Electron Count Rates (April 2018) | Cross-mission generalization |

---

## ⚙️ Physics-Informed Feature Engine (63 Variables)

Raw telemetry is transformed through domain physics to capture energy transfer into the magnetosphere:

* **Dynamic Solar Wind Pressure ($P_{dyn}$):**  
  $$P_{dyn} = 1.6726 \times 10^{-6} \cdot n_p \cdot v_{sw}^2$$
* **Interplanetary Convective Electric Field ($E_y$):**  
  $$E_y = -v_{sw} \cdot B_z \times 10^{-3}$$
* **Newell Magnetic Coupling Function ($d\Phi_{MP}/dt$):**  
  $$\frac{d\Phi_{MP}}{dt} = v_{sw}^{4/3} B_t^{2/3} \sin^{8/3}\left(\frac{\theta_c}{2}\right) \quad \text{where } \theta_c = \text{arctan2}(B_y, B_z)$$
* **Multi-Scale Temporal Integration:** Rolling means and standard deviations computed across **1h, 6h, 24h, and 72h** windows to capture both shock fronts and cumulative multi-day wave-particle acceleration.

---

## 🧠 Modeling Architecture & Performance

### 1. Dual-Model Framework
* **Gradient Boosted Trees (XGBoost):** Optimized for multi-scale tabular physics features, delivering fast inference and direct feature attribution via normalized Gini importance.
* **Deep Attention-BiGRU (PyTorch):** Ingests a sliding 72-step (6-hour) temporal history using Bidirectional GRU layers with dot-product self-attention to weight pre-storm substorm signatures.

### 2. Validation & Key Findings
* **Out-of-Sample Evaluation (2016 Test Set):** Outperformed persistence baselines across all horizons, achieving a **+10.6% error reduction at +24h** ($\text{MAE} \approx 0.19\text{ }\log_{10}\text{Flux}$).
* **Cross-Mission Transferability:** Validated against **ISRO GSAT-19 (GRASP)** telemetry at $48^\circ\text{ E}$, successfully reproducing diurnal orbital modulations and proving that L1-derived physical drivers transfer across different satellite longitudes and sensor designs.
* **Driver Attribution:** Gini importance rankings demonstrate that **72-hour integrated Newell coupling** and **convective electric field variance ($E_y$)** are the dominant predictors of sustained relativistic flux buildup.

---

## 🚀 Quick Start

### Python Environment (Linux / macOS / Windows)

```bash
# 1. Clone repository
git clone [https://github.com/anayekbote/space-radiation-forecast.git](https://github.com/anayekbote/space-radiation-forecast.git)
cd space-radiation-forecast

# 2. Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch interactive dashboard
streamlit run app.py
