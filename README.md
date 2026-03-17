# ChemEng Web — Streamlit Edition

Web version of the ChemEng Simulation Platform.
The original desktop app is unchanged at `../ChemEng_App`.

## Quick Start

```bash
cd ~/Desktop/Projects/ChemEng_Web

# Install dependencies (first time only)
pip install -r requirements.txt

# Run
streamlit run streamlit_app.py
```

The app opens automatically at **http://localhost:8501**

## Structure

```
ChemEng_Web/
├── streamlit_app.py        ← Home / dashboard
├── pages/                  ← One file per module (15 total)
│   ├── 01_🌡_Thermodynamics.py
│   ├── 02_⚗️_Reaction_Modeling.py
│   └── ...
├── app/controllers/        ← Shared backend (copied from desktop app)
├── core/                   ← Calculation engines (shared)
├── .streamlit/config.toml  ← Theme (blue accent, light background)
└── requirements.txt
```

## Modules

| # | Module | Tabs |
|---|--------|------|
| 1 | Thermodynamics | Ideal Gas, Vapor Pressure, VLE, EOS, H/S, Activity, Psychrometrics, Flame |
| 2 | Reaction Modeling | Reactors, Arrhenius, Series/Parallel, Sizing, Equilibrium, Non-Isothermal, RTD |
| 3 | Separation | McCabe-Thiele, Kremser, Flash, Extraction, Adsorption, Membrane |
| 4 | Optimization | LP, NLP, MILP, Dynamic Control |
| 5 | Heat Transfer | Flat/Composite/Cylinder wall, Convection, LMTD, NTU, Radiation |
| 6 | Fluid Dynamics | Pipe/Moody, Bernoulli, Orifice, Pump, Isentropic, Normal Shock |
| 7 | Database | Search, Vapor Pressure, Cp Curve, Compare |
| 8 | Process Control | FOPDT, 2nd Order, PID, Tuning, Bode |
| 9 | Machine Learning | Import CSV/Excel, Preprocess, Regression, Classification, Clustering, PCA, CV, Predict |
| 10 | Process Economics | Equipment Cost, CAPEX, OPEX, Cash Flow, Profitability, Sensitivity |
| 11 | Mass & Energy Balances | Stream, Mixer, Splitter, Material Balance, Energy Balance, Recycle, Composition |
| 12 | Safety & Risk | Dispersion, Explosion, Pool Fire, Risk Matrix, LOPA, Flammability |
| 13 | Bioprocess | Growth Kinetics, Batch, Chemostat, OTR, Sterilization |
| 14 | Polymer | MW Stats, Flory-Huggins, Mark-Houwink, Tg, WLF, Free-Radical |
| 15 | Electrochemistry | Nernst, Butler-Volmer, Faraday, Fuel Cell, Corrosion |
