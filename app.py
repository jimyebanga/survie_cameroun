"""
Analyse de Survie – Primo-nuptialité au Cameroun
Modèle : Gradient Boosting Survival Analysis (GBS)
Données : EDS Cameroun 2018
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings("ignore")

# ── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Survie Primo-nuptialité · Cameroun",
    page_icon="💍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Palette & variables */
:root {
    --vert-fonce:   #1B4332;
    --vert-moyen:   #2D6A4F;
    --vert-clair:   #52B788;
    --or:           #D4A017;
    --or-clair:     #F2C94C;
    --fond:         #F8F9FA;
    --blanc:        #FFFFFF;
    --gris-texte:   #2C3E50;
    --gris-leger:   #ECF0F1;
    --rouge:        #E74C3C;
    --bleu-stat:    #2980B9;
}

/* Fond général */
.main { background-color: var(--fond); }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Header hero */
.hero-banner {
    background: linear-gradient(135deg, var(--vert-fonce) 0%, var(--vert-moyen) 60%, var(--vert-clair) 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero-banner::after {
    content: "💍";
    position: absolute;
    right: 2.5rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.15;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero-sub {
    font-size: 1.05rem;
    opacity: 0.88;
    margin: 0;
    font-weight: 400;
}
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 20px;
    padding: 0.2rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-top: 1rem;
    margin-right: 0.5rem;
}

/* Cartes métriques */
.metric-card {
    background: var(--blanc);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    border-left: 5px solid var(--vert-clair);
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    margin-bottom: 1rem;
}
.metric-card.gold { border-left-color: var(--or); }
.metric-card.red  { border-left-color: var(--rouge); }
.metric-card.blue { border-left-color: var(--bleu-stat); }
.metric-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; color: #7f8c8d; font-weight: 600; margin-bottom: 0.3rem; }
.metric-value { font-size: 2rem; font-weight: 800; color: var(--gris-texte); line-height: 1; }
.metric-delta { font-size: 0.82rem; margin-top: 0.35rem; color: #95a5a6; }

/* Section title */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--vert-fonce);
    border-bottom: 3px solid var(--or);
    padding-bottom: 0.4rem;
    margin: 2rem 0 1.2rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--vert-fonce) 0%, #0d2b1f 100%);
}
[data-testid="stSidebar"] * { color: #ecf0f1 !important; }
[data-testid="stSidebar"] .stSlider > div > div { background: var(--vert-clair) !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--or-clair) !important; }

/* Bouton principal */
.stButton > button {
    background: linear-gradient(135deg, var(--vert-moyen), var(--vert-clair));
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.8rem;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.3px;
    transition: all 0.2s;
    box-shadow: 0 3px 10px rgba(45,106,79,0.35);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 5px 16px rgba(45,106,79,0.45);
}

/* Tableaux */
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.styled-table th {
    background: var(--vert-fonce);
    color: white;
    padding: 0.7rem 1rem;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.styled-table td { padding: 0.65rem 1rem; border-bottom: 1px solid var(--gris-leger); color: var(--gris-texte); }
.styled-table tr:nth-child(even) td { background: #f9fbfa; }
.styled-table tr:hover td { background: #eafaf1; }

/* Alerte info */
.info-box {
    background: #eafaf1;
    border-left: 4px solid var(--vert-clair);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    color: var(--vert-fonce);
    margin: 1rem 0;
}
.warn-box {
    background: #fef9e7;
    border-left: 4px solid var(--or);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    color: #7d6608;
    margin: 1rem 0;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.8rem;
    color: #95a5a6;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--gris-leger);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT & ENTRAÎNEMENT DU MODÈLE
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_and_train():
    """Charge les données, prépare et entraîne le modèle GBS."""
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.inspection import permutation_importance
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        from sksurv.util import Surv

        data = pd.read_csv("primo_nuptialite_clean.csv")
        data = data.dropna()

        cat_cols = ["milieu", "instruction", "richesse", "region", "religion"]
        df_ml = data.copy()
        le = LabelEncoder()
        for col in cat_cols:
            if col in df_ml.columns:
                df_ml[col] = le.fit_transform(df_ml[col].astype(str))

        X = df_ml.drop(columns=["duree", "union"])
        y = Surv.from_dataframe("union", "duree", df_ml)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        gbs = GradientBoostingSurvivalAnalysis(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
        gbs.fit(X_train, y_train)

        c_train = gbs.score(X_train, y_train)
        c_test  = gbs.score(X_test,  y_test)

        perm = permutation_importance(gbs, X_test, y_test, n_repeats=10, random_state=42)
        importances = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

        return {
            "model": gbs,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "c_train": c_train, "c_test": c_test,
            "importances": importances,
            "features": list(X.columns),
            "data": data,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "status": "ok",
        }
    except FileNotFoundError:
        return {"status": "no_data"}
    except ImportError as e:
        return {"status": "no_lib", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def generate_demo_data():
    """Génère des données de démonstration si le CSV n'est pas disponible."""
    np.random.seed(42)
    n = 3000
    df = pd.DataFrame({
        "duree": np.random.exponential(18, n).clip(1, 40).astype(int),
        "union": np.random.choice([True, False], n, p=[0.72, 0.28]),
        "milieu": np.random.choice(["Urbain", "Rural"], n),
        "instruction": np.random.choice(["Aucun", "Primaire", "Secondaire", "Supérieur"], n),
        "richesse": np.random.choice(["Pauvre", "Moyen", "Riche"], n),
        "region": np.random.choice(["Centre", "Littoral", "Nord", "Sud", "Est", "Ouest"], n),
        "religion": np.random.choice(["Chrétien", "Musulman", "Animiste", "Autre"], n),
        "age_menopause": np.random.randint(0, 2, n),
        "nb_enfants": np.random.randint(0, 8, n),
    })
    return df


@st.cache_resource(show_spinner=False)
def load_demo():
    """Entraîne un modèle GBS sur données de démonstration."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.inspection import permutation_importance
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.util import Surv

    data = generate_demo_data()
    df_ml = data.copy()
    cat_cols = ["milieu", "instruction", "richesse", "region", "religion"]
    le = LabelEncoder()
    for col in cat_cols:
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))

    X = df_ml.drop(columns=["duree", "union"])
    y = Surv.from_dataframe("union", "duree", df_ml)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    gbs = GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbs.fit(X_train, y_train)

    c_train = gbs.score(X_train, y_train)
    c_test  = gbs.score(X_test,  y_test)

    perm = permutation_importance(gbs, X_test, y_test, n_repeats=10, random_state=42)
    importances = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

    return {
        "model": gbs,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "c_train": c_train, "c_test": c_test,
        "importances": importances,
        "features": list(X.columns),
        "data": data,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "status": "demo",
    }


def c_index_label(c):
    if c > 0.70: return "🟢 Bonne discrimination", "#27ae60"
    if c > 0.60: return "🟡 Discrimination acceptable", "#f39c12"
    return "🔴 Discrimination faible", "#e74c3c"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎛️ Paramètres")
    st.markdown("---")

    st.markdown("### 📁 Données")
    uploaded = st.file_uploader(
        "Charger `primo_nuptialite_clean.csv`",
        type=["csv"],
        help="Fichier CSV avec colonnes : duree, union, milieu, instruction, richesse, region, religion…"
    )
    use_demo = st.checkbox("Utiliser les données de démo", value=True if not uploaded else False)

    st.markdown("---")
    st.markdown("### ⚙️ Modèle GBS")
    n_estimators = st.slider("Nombre d'arbres", 50, 300, 100, 25)
    learning_rate = st.select_slider("Taux d'apprentissage", [0.01, 0.05, 0.1, 0.15, 0.2], value=0.1)
    max_depth = st.slider("Profondeur max", 1, 6, 3)
    test_size = st.slider("Taille du jeu test (%)", 10, 40, 20, 5)

    retrain = st.button("🔄 Ré-entraîner le modèle", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Visualisation")
    n_individus = st.slider("Individus pour courbes de survie", 3, 15, 5)
    show_ci = st.checkbox("Afficher intervalle de confiance", value=False)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; opacity:0.6; text-align:center; line-height:1.6'>
    GBS · scikit-survival<br>
    EDS Cameroun 2018<br>
    Primo-nuptialité
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════
# Vérifier si sksurv est disponible
try:
    import sksurv
    sksurv_available = True
except ImportError:
    sksurv_available = False

if not sksurv_available:
    st.error("""
    ### ⚠️ Dépendance manquante : `scikit-survival`
    
    Installez le package avec :
    ```
    pip install scikit-survival
    ```
    Puis relancez l'application.
    """)
    st.stop()

# Chargement / entraînement
if uploaded:
    with st.spinner("Chargement et entraînement en cours…"):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.inspection import permutation_importance
            from sksurv.ensemble import GradientBoostingSurvivalAnalysis
            from sksurv.util import Surv

            data = pd.read_csv(uploaded)
            data = data.dropna()
            cat_cols = [c for c in ["milieu", "instruction", "richesse", "region", "religion"] if c in data.columns]
            df_ml = data.copy()
            le = LabelEncoder()
            for col in cat_cols:
                df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            X = df_ml.drop(columns=["duree", "union"])
            y = Surv.from_dataframe("union", "duree", df_ml)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
            gbs = GradientBoostingSurvivalAnalysis(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
            gbs.fit(X_train, y_train)
            c_train = gbs.score(X_train, y_train)
            c_test  = gbs.score(X_test,  y_test)
            perm = permutation_importance(gbs, X_test, y_test, n_repeats=10, random_state=42)
            importances = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
            state = {
                "model": gbs, "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
                "c_train": c_train, "c_test": c_test,
                "importances": importances, "features": list(X.columns),
                "data": data, "n_train": len(X_train), "n_test": len(X_test), "status": "ok",
            }
        except Exception as e:
            state = {"status": "error", "error": str(e)}
elif use_demo:
    with st.spinner("Chargement du modèle de démonstration…"):
        state = load_demo()
else:
    state = load_and_train()
    if state["status"] == "no_data" and use_demo:
        state = load_demo()


# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
demo_badge = '<span class="badge">⚠️ Données de démo</span>' if state.get("status") == "demo" else '<span class="badge">✅ Données réelles</span>'
st.markdown(f"""
<div class="hero-banner">
  <p class="hero-title">Analyse de Survie · Primo-nuptialité</p>
  <p class="hero-sub">Gradient Boosting Survival Analysis — EDS Cameroun 2018</p>
  <span class="badge">🌿 Cameroun</span>
  <span class="badge">📐 GBS Model</span>
  {demo_badge}
</div>
""", unsafe_allow_html=True)

# Vérification état
if state["status"] == "error":
    st.error(f"**Erreur lors de l'entraînement :** {state.get('error', 'Inconnue')}")
    st.stop()
if state["status"] == "no_lib":
    st.error(f"**Bibliothèque manquante :** {state.get('error', '')}")
    st.stop()

if state.get("status") == "demo":
    st.markdown("""
    <div class="warn-box">
    ⚠️ <strong>Mode démonstration</strong> — Les données affichées sont synthétiques.
    Chargez <code>primo_nuptialite_clean.csv</code> dans la barre latérale pour utiliser les vraies données EDS 2018.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performances",
    "🔍 Importance des variables",
    "📉 Courbes de survie",
    "🔮 Prédiction individuelle",
    "📋 Données & Export",
])


# ─── TAB 1 : PERFORMANCES ─────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">📊 Performances du modèle GBS</div>', unsafe_allow_html=True)

    c_train = state["c_train"]
    c_test  = state["c_test"]
    ecart   = c_train - c_test
    label_test, col_test = c_index_label(c_test)

    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">C-index · Train</div>
          <div class="metric-value">{c_train:.4f}</div>
          <div class="metric-delta">Pouvoir discriminant en train</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card gold">
          <div class="metric-label">C-index · Test</div>
          <div class="metric-value">{c_test:.4f}</div>
          <div class="metric-delta">{label_test}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        color_class = "red" if ecart > 0.08 else "metric-card"
        st.markdown(f"""
        <div class="{color_class} metric-card">
          <div class="metric-label">Écart (surfit)</div>
          <div class="metric-value">{ecart:.4f}</div>
          <div class="metric-delta">{"⚠️ Surapprentissage probable" if ecart > 0.08 else "✅ Généralisation correcte"}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card blue">
          <div class="metric-label">Observations</div>
          <div class="metric-value">{state['n_train'] + state['n_test']:,}</div>
          <div class="metric-delta">Train {state['n_train']:,} · Test {state['n_test']:,}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Graphique jauges
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=c_test,
            delta={"reference": c_train, "valueformat": ".4f", "prefix": "vs train: "},
            title={"text": "C-index Test", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0.4, 1.0], "tickwidth": 1},
                "bar": {"color": "#2D6A4F"},
                "steps": [
                    {"range": [0.4, 0.6], "color": "#fadbd8"},
                    {"range": [0.6, 0.7], "color": "#fdebd0"},
                    {"range": [0.7, 1.0], "color": "#d5f5e3"},
                ],
                "threshold": {"line": {"color": "#D4A017", "width": 3}, "thickness": 0.75, "value": 0.7},
            },
            number={"valueformat": ".4f", "font": {"size": 30}},
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=60, b=20, l=20, r=20), paper_bgcolor="white")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_g2:
        # Barchart Train vs Test
        models_cmp = ["Train", "Test"]
        values_cmp = [c_train, c_test]
        colors_cmp = ["#52B788", "#D4A017"]

        fig_bar = go.Figure(go.Bar(
            x=models_cmp, y=values_cmp,
            marker=dict(color=colors_cmp, line=dict(color="#1B4332", width=1.5)),
            text=[f"{v:.4f}" for v in values_cmp],
            textposition="outside",
            textfont=dict(size=13, color="#2C3E50"),
        ))
        fig_bar.add_hline(y=0.5, line_dash="dash", line_color="#e74c3c", annotation_text="Aléatoire (0.5)", annotation_position="bottom right")
        fig_bar.add_hline(y=0.7, line_dash="dot",  line_color="#27ae60", annotation_text="Seuil bon (0.7)", annotation_position="bottom right")
        fig_bar.update_layout(
            title="C-index : Train vs Test",
            yaxis=dict(range=[0.4, 1.05], title="C-index"),
            xaxis=dict(title=""),
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=300,
            margin=dict(t=50, b=30, l=40, r=30),
            font=dict(family="sans-serif"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Interprétation
    st.markdown(f"""
    <div class="info-box">
    <strong>📌 Interprétation :</strong> Le C-index (concordance index) mesure la capacité du modèle à ordonner correctement les durées avant union.
    Un C-index de <strong>{c_test:.4f}</strong> signifie que dans {c_test*100:.1f}% des paires comparées, le modèle prédit correctement
    laquelle des deux femmes entrera en union en premier.<br><br>
    Le modèle <strong>GBS</strong> (Gradient Boosting Survival) combine des arbres de décision en séquence,
    chaque arbre corrigeant les erreurs du précédent. Il est particulièrement adapté aux données de survie avec censure.
    </div>
    """, unsafe_allow_html=True)

    # Paramètres utilisés
    st.markdown('<div class="section-title">⚙️ Paramètres du modèle</div>', unsafe_allow_html=True)
    params_df = pd.DataFrame({
        "Paramètre": ["n_estimators", "learning_rate", "max_depth", "random_state", "Test size"],
        "Valeur": [n_estimators, learning_rate, max_depth, 42, f"{test_size}%"],
        "Description": [
            "Nombre d'arbres de décision",
            "Taux d'apprentissage (shrinkage)",
            "Profondeur maximale de chaque arbre",
            "Graine pour la reproductibilité",
            "Proportion des données de test",
        ]
    })
    st.dataframe(params_df, use_container_width=True, hide_index=True)


# ─── TAB 2 : IMPORTANCE DES VARIABLES ─────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">🔍 Importance des variables (Permutation)</div>', unsafe_allow_html=True)

    importances = state["importances"]

    col_i1, col_i2 = st.columns([2, 1])

    with col_i1:
        # Horizontal bar chart
        imp_sorted = importances.sort_values(ascending=True)
        colors_imp = ["#52B788" if v >= 0 else "#e74c3c" for v in imp_sorted.values]

        fig_imp = go.Figure(go.Bar(
            x=imp_sorted.values,
            y=imp_sorted.index,
            orientation="h",
            marker=dict(color=colors_imp, line=dict(color="#1B4332", width=0.8)),
            text=[f"{v:.4f}" for v in imp_sorted.values],
            textposition="outside",
        ))
        fig_imp.update_layout(
            title="Importance par permutation – GBS",
            xaxis=dict(title="Réduction du C-index", zeroline=True, zerolinecolor="#ccc"),
            yaxis=dict(title=""),
            height=max(350, len(importances) * 45),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=50, b=30, l=10, r=60),
            font=dict(size=12),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_i2:
        st.markdown("**Rang des variables**")
        imp_display = pd.DataFrame({
            "Variable": importances.index,
            "Importance": importances.values.round(5),
            "Rang": range(1, len(importances)+1)
        })
        # Barre de progression HTML
        max_imp = imp_display["Importance"].abs().max()
        st.dataframe(
            imp_display[["Rang", "Variable", "Importance"]],
            use_container_width=True,
            hide_index=True,
        )

        # Top variable
        top_var = importances.idxmax()
        top_val = importances.max()
        st.markdown(f"""
        <div class="info-box" style="margin-top:1rem">
        🏆 <strong>Variable la plus importante :</strong><br>
        <code>{top_var}</code> (importance = {top_val:.4f})
        </div>
        """, unsafe_allow_html=True)

    # Treemap
    st.markdown('<div class="section-title">🗂️ Vue globale (treemap)</div>', unsafe_allow_html=True)
    imp_pos = importances[importances > 0]
    if len(imp_pos) > 0:
        fig_tree = px.treemap(
            names=imp_pos.index.tolist(),
            values=imp_pos.values.tolist(),
            title="Poids relatif des variables (importance positive)",
            color=imp_pos.values.tolist(),
            color_continuous_scale=["#d5f5e3", "#2D6A4F"],
        )
        fig_tree.update_layout(height=350, paper_bgcolor="white", margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_tree, use_container_width=True)


# ─── TAB 3 : COURBES DE SURVIE ─────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">📉 Courbes de survie prédites (GBS)</div>', unsafe_allow_html=True)

    gbs_model = state["model"]
    X_test    = state["X_test"]

    n_ind = min(n_individus, len(X_test))
    np.random.seed(7)
    idxs = np.random.choice(len(X_test), n_ind, replace=False)
    X_sample = X_test.iloc[idxs]

    surv_funcs = gbs_model.predict_survival_function(X_sample)

    # Palette verte→or
    palette = px.colors.sample_colorscale("Viridis", np.linspace(0.1, 0.9, n_ind))

    fig_surv = go.Figure()
    for i, sf in enumerate(surv_funcs):
        times = sf.x
        probs = sf(times)
        fig_surv.add_trace(go.Scatter(
            x=times, y=probs,
            mode="lines",
            name=f"Individu {i+1}",
            line=dict(width=2.5, color=palette[i]),
            hovertemplate=f"<b>Individu {i+1}</b><br>Âge: %{{x:.1f}}<br>S(t): %{{y:.3f}}<extra></extra>",
        ))

    fig_surv.add_hline(y=0.5, line_dash="dash", line_color="#e74c3c", line_width=1.5,
                       annotation_text="S(t) = 0.5 (médiane)", annotation_position="bottom right")
    fig_surv.update_layout(
        title="Probabilité de ne pas être en union (GBS)",
        xaxis=dict(title="Âge (années)", showgrid=True, gridcolor="#ecf0f1"),
        yaxis=dict(title="S(t) – Probabilité de survie", range=[0, 1.05], showgrid=True, gridcolor="#ecf0f1"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=460,
        legend=dict(orientation="v", x=1.02, y=0.95),
        margin=dict(t=60, b=50, l=60, r=140),
        hovermode="x unified",
        font=dict(size=12),
    )
    st.plotly_chart(fig_surv, use_container_width=True)

    # Tableau des probabilités aux points clés
    st.markdown('<div class="section-title">📊 Probabilités aux âges clés</div>', unsafe_allow_html=True)

    key_ages = [15, 18, 20, 25, 30]
    rows = []
    for i, sf in enumerate(surv_funcs):
        row = {"Individu": f"Individu {i+1}"}
        for age in key_ages:
            idx_age = np.searchsorted(sf.x, age)
            if idx_age < len(sf.x):
                row[f"Âge {age}"] = f"{sf(sf.x[idx_age]):.3f}"
            else:
                row[f"Âge {age}"] = "—"
        rows.append(row)

    prob_df = pd.DataFrame(rows)
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="info-box">
    📌 <strong>Lecture :</strong> S(t) représente la probabilité qu'une femme <em>ne soit pas encore entrée en union</em> à l'âge t.
    Une courbe qui descend rapidement indique un risque élevé d'union précoce. La ligne rouge (S=0.5) marque l'âge médian d'entrée en union pour chaque profil.
    </div>
    """, unsafe_allow_html=True)


# ─── TAB 4 : PRÉDICTION INDIVIDUELLE ──────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">🔮 Prédiction pour un profil individuel</div>', unsafe_allow_html=True)

    features = state["features"]
    X_test   = state["X_test"]

    st.markdown("Définissez le profil d'un individu pour obtenir sa courbe de survie personnalisée.")

    # Créer formulaire en colonnes
    col_f1, col_f2 = st.columns(2)
    user_input = {}

    # Valeurs par défaut tirées des données
    defaults = dict(X_test.iloc[0])

    feature_labels = {
        "milieu": ("Milieu de résidence", {"Urbain": 1, "Rural": 0}),
        "instruction": ("Niveau d'instruction", {"Aucun": 0, "Primaire": 1, "Secondaire": 2, "Supérieur": 3}),
        "richesse": ("Quintile de richesse", {"Pauvre": 0, "Moyen": 1, "Riche": 2}),
        "region": ("Région", {"Centre": 0, "Littoral": 1, "Nord": 2, "Sud": 3, "Est": 4, "Ouest": 5}),
        "religion": ("Religion", {"Chrétien": 0, "Musulman": 1, "Animiste": 2, "Autre": 3}),
    }

    for i, feat in enumerate(features):
        col = col_f1 if i % 2 == 0 else col_f2
        with col:
            if feat in feature_labels:
                label, options = feature_labels[feat]
                selected = st.selectbox(
                    label,
                    list(options.keys()),
                    index=0,
                    key=f"feat_{feat}"
                )
                user_input[feat] = options[selected]
            else:
                default_val = float(defaults.get(feat, 0))
                user_input[feat] = st.number_input(
                    feat.replace("_", " ").title(),
                    value=default_val,
                    step=1.0,
                    key=f"feat_{feat}"
                )

    st.markdown("---")

    if st.button("🔮 Prédire la courbe de survie", use_container_width=True):
        with st.spinner("Calcul en cours…"):
            try:
                input_df = pd.DataFrame([user_input], columns=features)
                sf_pred  = state["model"].predict_survival_function(input_df)[0]

                # Trouver la médiane (S(t) = 0.5)
                median_idx = np.searchsorted(-sf_pred(sf_pred.x), -0.5)
                median_age = sf_pred.x[median_idx] if median_idx < len(sf_pred.x) else None

                col_r1, col_r2 = st.columns([2, 1])
                with col_r1:
                    fig_ind = go.Figure()
                    fig_ind.add_trace(go.Scatter(
                        x=sf_pred.x, y=sf_pred(sf_pred.x),
                        fill="tozeroy",
                        fillcolor="rgba(82, 183, 136, 0.15)",
                        line=dict(color="#2D6A4F", width=3),
                        name="Profil individuel",
                        hovertemplate="Âge: %{x:.1f}<br>S(t): %{y:.3f}<extra></extra>",
                    ))
                    if median_age:
                        fig_ind.add_vline(x=median_age, line_dash="dot", line_color="#D4A017", line_width=2,
                                          annotation_text=f"Âge médian : {median_age:.1f} ans",
                                          annotation_position="top right")
                    fig_ind.add_hline(y=0.5, line_dash="dash", line_color="#e74c3c", line_width=1.5)
                    fig_ind.update_layout(
                        title="Courbe de survie personnalisée",
                        xaxis=dict(title="Âge (années)", showgrid=True, gridcolor="#ecf0f1"),
                        yaxis=dict(title="S(t)", range=[0, 1.05], showgrid=True, gridcolor="#ecf0f1"),
                        paper_bgcolor="white", plot_bgcolor="white",
                        height=380, margin=dict(t=50, b=40, l=50, r=30),
                        font=dict(size=12),
                    )
                    st.plotly_chart(fig_ind, use_container_width=True)

                with col_r2:
                    st.markdown("**Résumé du profil**")
                    probs_key = {}
                    for age in [15, 18, 20, 25, 30]:
                        idx_a = np.searchsorted(sf_pred.x, age)
                        probs_key[f"Âge {age} ans"] = f"{sf_pred(sf_pred.x[min(idx_a, len(sf_pred.x)-1)]):.3f}" if idx_a < len(sf_pred.x) else "—"

                    if median_age:
                        st.markdown(f"""
                        <div class="metric-card gold">
                          <div class="metric-label">Âge médian d'union</div>
                          <div class="metric-value">{median_age:.1f} ans</div>
                          <div class="metric-delta">S(t) = 0.5</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("**P(pas en union) par âge :**")
                    for k, v in probs_key.items():
                        st.markdown(f"- **{k}** → `{v}`")

            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")


# ─── TAB 5 : DONNÉES & EXPORT ──────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">📋 Données brutes</div>', unsafe_allow_html=True)

    data = state["data"]

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Observations totales</div>
          <div class="metric-value">{len(data):,}</div>
        </div>""", unsafe_allow_html=True)
    with col_d2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Variables</div>
          <div class="metric-value">{data.shape[1]}</div>
        </div>""", unsafe_allow_html=True)
    with col_d3:
        if "union" in data.columns:
            pct_union = data["union"].mean() * 100
            st.markdown(f"""
            <div class="metric-card gold">
              <div class="metric-label">% entrées en union</div>
              <div class="metric-value">{pct_union:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    st.dataframe(data.head(100), use_container_width=True)

    st.markdown('<div class="section-title">📤 Export des résultats</div>', unsafe_allow_html=True)

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        # Export importance
        imp_csv = state["importances"].reset_index()
        imp_csv.columns = ["Variable", "Importance"]
        csv_imp = imp_csv.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Télécharger l'importance des variables (CSV)",
            csv_imp,
            file_name="gbs_importance_variables.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_e2:
        # Export résumé performances
        resume = pd.DataFrame({
            "Modèle": ["Gradient Boosting Survival (GBS)"],
            "C-index Train": [round(state["c_train"], 4)],
            "C-index Test":  [round(state["c_test"],  4)],
            "Écart":         [round(state["c_train"] - state["c_test"], 4)],
            "n_estimators":  [n_estimators],
            "learning_rate": [learning_rate],
            "max_depth":     [max_depth],
        })
        csv_res = resume.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Télécharger le résumé des performances (CSV)",
            csv_res,
            file_name="gbs_performances.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Statistiques descriptives
    st.markdown('<div class="section-title">📊 Statistiques descriptives</div>', unsafe_allow_html=True)
    st.dataframe(data.describe().round(3), use_container_width=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  💍 Analyse de Survie · Primo-nuptialité au Cameroun · EDS 2018 &nbsp;|&nbsp;
  Modèle : <strong>Gradient Boosting Survival Analysis</strong> (scikit-survival) &nbsp;|&nbsp;
  Streamlit · Python
</div>
""", unsafe_allow_html=True)
