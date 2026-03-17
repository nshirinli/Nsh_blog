"""ChemEng Web — Home / Dashboard."""
import streamlit as st

st.set_page_config(
    page_title="ChemEng Simulation Platform",
    page_icon="⚗️",
    layout="wide",
)

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #1a1e30 0%, #23284a 100%);
    border-radius: 12px;
    padding: 36px 40px 28px 40px;
    margin-bottom: 24px;
}
.hero h1 { color: #ffffff; font-size: 2rem; margin: 0 0 8px 0; }
.hero p  { color: #8fa3c8; font-size: 1rem; margin: 0 0 18px 0; }
.chip {
    display: inline-block;
    background: rgba(255,255,255,0.10);
    color: #c4d3f0;
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.82rem;
    margin-right: 8px;
}
.mod-card {
    background: #ffffff;
    border-radius: 10px;
    border-left: 4px solid var(--accent);
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    transition: box-shadow .15s;
}
.mod-title { font-weight: 700; font-size: 0.95rem; color: #111827; }
.mod-desc  { font-size: 0.82rem; color: #6b7280; margin-top: 3px; }
</style>

<div class="hero">
  <h1>⚗️  ChemEng Simulation Platform</h1>
  <p>A modular web application for chemical engineering calculations, simulations, process design, and data analysis.</p>
  <span class="chip">15 Modules</span>
  <span class="chip">Interactive Plots</span>
  <span class="chip">Offline Ready</span>
</div>
""", unsafe_allow_html=True)

MODULES = [
    ("🌡", "Thermodynamics",         "Ideal gas law, vapor pressure, VLE, real-gas EOS, enthalpy/entropy, psychrometrics, and adiabatic flame.", "#ef4444"),
    ("⚗️", "Reaction Modeling",       "Batch, CSTR, PFR kinetics, Arrhenius, series/parallel reactions, equilibrium, non-isothermal reactors, and RTD.", "#f97316"),
    ("🔬", "Separation",              "McCabe-Thiele distillation, Kremser absorption, Rachford-Rice flash, extraction, adsorption, and membrane.", "#eab308"),
    ("📈", "Optimization",            "Linear, nonlinear, mixed-integer, and dynamic optimal control problems with constraint handling.", "#22c55e"),
    ("🔥", "Heat Transfer",           "Conduction (flat/composite/cylinder), convection, heat exchangers (LMTD/NTU), and radiation.", "#f97316"),
    ("💧", "Fluid Dynamics",          "Pipe flow & Moody chart, Bernoulli balance, pump sizing, compressible flow, and normal shocks.", "#3b82f6"),
    ("🗄️", "Database",               "Search 40+ compounds. View physical, critical, Antoine & Cp properties. Compare & plot.", "#8b5cf6"),
    ("📊", "Process Control",         "FOPDT dynamics, PID simulation, ZN/ITAE/IMC tuning, and Bode frequency response.", "#06b6d4"),
    ("🤖", "Machine Learning",        "Import CSV/Excel, preprocess, train regression/classification/clustering/PCA, cross-validate, predict.", "#ec4899"),
    ("💰", "Process Economics",       "Equipment cost (Turton), CAPEX (bare-module/Lang), OPEX, cash flow, NPV/IRR/payback, sensitivity.", "#f59e0b"),
    ("⚖️", "Mass & Energy Balances", "Stream properties, mixer, splitter, material balance, energy balance, recycle loop, composition converter.", "#10b981"),
    ("🦺", "Safety & Risk Analysis",  "Gaussian plume, VCE (TNT), pool fire, 5×5 risk matrix, LOPA, and flammability limits.", "#ef4444"),
    ("🧬", "Bioprocess Engineering",  "Monod/Andrews growth kinetics, batch bioreactor ODE, chemostat, oxygen transfer, thermal sterilization.", "#84cc16"),
    ("🧪", "Polymer Engineering",     "MW statistics (Mn, Mw, PDI), Flory-Huggins, Mark-Houwink, glass transition (Fox/WLF), free-radical kinetics.", "#a855f7"),
    ("⚡", "Electrochemistry",        "Nernst equation, Butler-Volmer i–η curve, Faraday electrolysis, fuel cell polarization, corrosion rate.", "#eab308"),
]

cols = st.columns(3)
for i, (icon, title, desc, color) in enumerate(MODULES):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="mod-card" style="--accent:{color}">
          <div class="mod-title">{icon}  {title}</div>
          <div class="mod-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Use the sidebar to navigate between modules.  |  ChemEng Web v1.0.0")
