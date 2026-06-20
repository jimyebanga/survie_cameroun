import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

st.set_page_config(
    page_title="Primo-nuptialité · Cameroun",
    page_icon="🇨🇲",
    layout="wide"
)

st.markdown("""
<style>
/* ── Palette ── */
:root {
    --vert:   #007A5E;
    --jaune:  #FCD116;
    --rouge:  #CE1126;
    --dark:   #0f2318;
    --texte:  #1a1a2e;
    --gris:   #f5f6fa;
}

/* Fond */
.main { background: var(--gris); }
.block-container { padding: 1.2rem 2rem 2rem 2rem; }

/* ── Bandeau tricolore ── */
.hero {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.14);
}
.hero-stripe { padding: 1.4rem 1.6rem; color: white; }
.hero-stripe.v { background: var(--vert); }
.hero-stripe.j { background: var(--jaune); color: var(--texte); }
.hero-stripe.r { background: var(--rouge); text-align: right; }
.hero-title { font-size: 1.45rem; font-weight: 800; margin: 0; line-height: 1.2; }
.hero-sub   { font-size: 0.82rem; opacity: 0.85; margin: 0.25rem 0 0; }

/* ── Cartes stat ── */
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; }
.stat-card {
    flex: 1;
    background: white;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    border-bottom: 4px solid var(--vert);
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    text-align: center;
}
.stat-card.j { border-bottom-color: var(--jaune); }
.stat-card.r { border-bottom-color: var(--rouge); }
.stat-val   { font-size: 1.7rem; font-weight: 800; color: var(--texte); }
.stat-label { font-size: 0.78rem; color: #777; margin-top: 0.15rem; text-transform: uppercase; letter-spacing: .5px; }

/* ── Résultat proba ── */
.prob-grid { display: flex; gap: 0.8rem; margin: 0.8rem 0; }
.prob-box {
    flex: 1; text-align: center;
    background: white;
    border-radius: 8px;
    padding: 0.9rem 0.5rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
}
.prob-box .p-age  { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: .4px; }
.prob-box .p-val  { font-size: 1.6rem; font-weight: 800; color: var(--vert); }

/* ── Interprétation ── */
.interp {
    border-radius: 8px;
    padding: 0.85rem 1.2rem;
    font-size: 0.92rem;
    margin: 0.8rem 0 1.1rem;
    font-weight: 600;
}
.interp.high { background: #e6f4f0; border-left: 5px solid var(--vert); color: #004d3a; }
.interp.mod  { background: #fffae6; border-left: 5px solid var(--jaune); color: #6b5400; }
.interp.low  { background: #fdecea; border-left: 5px solid var(--rouge); color: #8b0000; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--dark) !important;
}
[data-testid="stSidebar"] * { color: #e8f5f0 !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--jaune) !important; }
[data-testid="stSidebar"] .stButton > button {
    background: var(--vert) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    width: 100% !important;
    margin-top: 0.5rem;
}

/* ── Section titles ── */
.section-head {
    font-size: 1rem;
    font-weight: 700;
    color: var(--vert);
    border-left: 4px solid var(--jaune);
    padding-left: 0.6rem;
    margin: 1.2rem 0 0.7rem;
}

/* ── Footer ── */
.footer { text-align: center; font-size: 0.77rem; color: #aaa; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ── Chargement modèle ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("gbs_model.pkl"):
        with open("gbs_model.pkl", "rb") as f:
            model, scaler, cols = pickle.load(f)
        return model, scaler, cols
    return None, None, None

@st.cache_data
def load_data():
    if os.path.exists("primo_nuptialite_clean.csv"):
        return pd.read_csv("primo_nuptialite_clean.csv")
    return None


# ── Courbe matplotlib (légère, pas de plotly) ────────────────────────────────
def plot_curve(surv_func, label):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    times = surv_func.x
    probs = surv_func(times)

    ax.step(times, probs, where="post", color="#007A5E", linewidth=2.8, label=label)
    ax.fill_between(times, 0, probs, step="post", alpha=0.12, color="#007A5E")
    ax.axhline(0.5, color="#CE1126", linestyle="--", linewidth=1.3, alpha=0.7, label="S(t) = 0.5")

    ax.set_xlabel("Âge (années)", fontsize=10, color="#444")
    ax.set_ylabel("Probabilité de survie S(t)", fontsize=10, color="#444")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.15, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-stripe v">
    <p class="hero-title">Primo-nuptialité</p>
    <p class="hero-sub">Cameroun · EDS 2018</p>
  </div>
  <div class="hero-stripe j">
    <p class="hero-title">Analyse de survie</p>
    <p class="hero-sub">Gradient Boosting Survival</p>
  </div>
  <div class="hero-stripe r">
    <p class="hero-title">Prédiction</p>
    <p class="hero-sub">C-index test : 0.6762</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Chargement ───────────────────────────────────────────────────────────────
model, scaler, cols = load_model()
data = load_data()

if model is None or data is None:
    st.error("⚠️ Fichiers manquants : placez `gbs_model.pkl` et `primo_nuptialite_clean.csv` à la racine du projet.")
    st.stop()


# ── Stats rapides ─────────────────────────────────────────────────────────────
age_moyen = data[data["union"] == 1]["duree"].mean()
st.markdown(f"""
<div class="stat-row">
  <div class="stat-card">
    <div class="stat-val">{len(data):,}</div>
    <div class="stat-label">Individus</div>
  </div>
  <div class="stat-card j">
    <div class="stat-val">{data['union'].mean():.1%}</div>
    <div class="stat-label">Taux d'union observé</div>
  </div>
  <div class="stat-card r">
    <div class="stat-val">{age_moyen:.1f} ans</div>
    <div class="stat-label">Âge moyen d'entrée en union</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Profil individuel")

    milieu      = st.selectbox("Milieu", ["Urbain", "Rural"])
    instruction = st.selectbox("Instruction", ["Aucun", "Primaire", "Secondaire", "Superieur"])
    region      = st.selectbox("Région", ["Adamaoua", "Centre", "Est", "Extreme-Nord", "Littoral",
                                           "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest"])
    religion    = st.selectbox("Religion", ["Catholique", "Protestant", "Musulman", "Animiste", "Autre"])
    richesse    = st.selectbox("Richesse", ["Pauvre", "Moyen", "Riche"])

    st.markdown("---")
    predict = st.button("🔮 Calculer la courbe", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; opacity:0.5; line-height:1.7'>
    Modèle GBS · scikit-survival<br>
    EDS Cameroun 2018
    </div>
    """, unsafe_allow_html=True)


# ── Résultats ─────────────────────────────────────────────────────────────────
if predict:
    vals = {
        "milieu":       0 if milieu == "Urbain" else 1,
        "instruction":  ["Aucun", "Primaire", "Secondaire", "Superieur"].index(instruction),
        "region":       ["Adamaoua", "Centre", "Est", "Extreme-Nord", "Littoral",
                         "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest"].index(region),
        "religion":     ["Catholique", "Protestant", "Musulman", "Animiste", "Autre"].index(religion),
        "richesse":     ["Pauvre", "Moyen", "Riche"].index(richesse),
    }

    X = pd.DataFrame([vals])
    # Réordonner selon cols du modèle si nécessaire
    if cols is not None:
        for c in cols:
            if c not in X.columns:
                X[c] = 0
        X = X[cols]

    X_scaled = scaler.transform(X)
    surv     = model.predict_survival_function(X_scaled)[0]

    def get_p(age):
        idx = np.searchsorted(surv.x, age)
        if idx >= len(surv.x): return float(surv(surv.x[-1]))
        return float(surv(surv.x[idx]))

    # Probas aux âges clés
    st.markdown('<div class="section-head">📊 Probabilités de ne pas être en union</div>', unsafe_allow_html=True)
    ages_cles = [20, 25, 30, 35, 40]
    boxes = "".join(
        f'<div class="prob-box"><div class="p-age">{a} ans</div>'
        f'<div class="p-val">{get_p(a):.1%}</div></div>'
        for a in ages_cles
    )
    st.markdown(f'<div class="prob-grid">{boxes}</div>', unsafe_allow_html=True)

    # Interprétation
    p30 = get_p(30)
    if p30 > 0.5:
        cls, msg = "high", "Profil à faible risque d'union précoce — probabilité élevée de rester célibataire à 30 ans."
    elif p30 > 0.25:
        cls, msg = "mod",  "Profil à risque modéré — une femme sur quatre à trois sur quatre sera en union à 30 ans."
    else:
        cls, msg = "low",  "Profil à risque élevé d'union précoce — très forte probabilité d'être en union avant 30 ans."
    st.markdown(f'<div class="interp {cls}">🔍 {msg}</div>', unsafe_allow_html=True)

    # Courbe
    st.markdown('<div class="section-head">📉 Courbe de survie</div>', unsafe_allow_html=True)
    fig = plot_curve(surv, f"{milieu} · {instruction} · {region}")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Détails dans un expander
    with st.expander("Tableau complet des probabilités"):
        ages_detail = [15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 45, 50]
        df_detail = pd.DataFrame({
            "Âge (ans)": ages_detail,
            "P(pas en union)": [f"{get_p(a):.3f}" for a in ages_detail],
            "P(en union)": [f"{1 - get_p(a):.3f}" for a in ages_detail],
        })
        st.dataframe(df_detail, hide_index=True, use_container_width=True)

else:
    st.info("👈 Définissez un profil dans la barre latérale et cliquez **Calculer la courbe**.")


# ── À propos ──────────────────────────────────────────────────────────────────
with st.expander("À propos du modèle"):
    st.markdown("""
    | Paramètre | Valeur |
    |-----------|--------|
    | Modèle | Gradient Boosting Survival Analysis |
    | Variables | Milieu, Instruction, Région, Religion, Richesse |
    | C-index Train | 0.6824 |
    | C-index Test | 0.6762 |
    | Source | EDS Cameroun 2018 |
    """)

st.markdown('<div class="footer">Analyse de primo-nuptialité · Cameroun · EDS 2018</div>', unsafe_allow_html=True)