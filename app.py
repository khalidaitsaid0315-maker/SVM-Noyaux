import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# === AUTO-ENTRAÎNEMENT SI LES MODÈLES MANQUENT (pour Streamlit Cloud) ===
if not os.path.exists('models/saved_models/all_results.pkl'):
    st.warning("Modèles non trouvés → Entraînement automatique en cours (première fois seulement)...")
    import subprocess
    # Utilisation de sys.executable pour garantir l'accès aux librairies installées
    result = subprocess.run([sys.executable, "train_svm.py"], capture_output=True, text=True)
    
    if result.returncode == 0:
        st.success("Entraînement terminé ! L'app est prête.")
    else:
        st.error("Échec de l'entraînement automatique.")
        st.code(result.stdout + result.stderr)
        st.stop()

# =============================================
# CONFIGURATION DE LA PAGE
# =============================================
st.set_page_config(
    page_title="SVM à Noyaux - Détection de Cancer",
    page_icon="DNA",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {padding: 0 1rem;}
    .block-container {padding-top: 2rem;}

    /* Header principal */
    .medical-header {
        background: linear-gradient(135deg, #006666, #009999);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,102,102,0.3);
        margin-bottom: 3rem;
    }
    .medical-header h1 {font-size: 3.2rem; font-weight: 900; margin: 0;}

    /* Résultat prédiction */
    .prediction-box {
        padding: 3.5rem; border-radius: 25px; text-align: center;
        font-size: 3rem; font-weight: 900; color: white;
        margin: 3rem 0; box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    }
    .benin {background: linear-gradient(135deg, #11998e, #38ef7d);}
    .malin {background: linear-gradient(135deg, #eb3349, #f45c43);}

    /* Boutons */
    .stButton>button {
        background: #006666 !important; color: white !important;
        border-radius: 50px !important; height: 3.8rem !important;
        font-size: 1.4rem !important; font-weight: bold !important;
    }

    /* SIDEBAR EN VERT ÉLÉGANT */
    [data-testid="stSidebar"] {
        background-color: #f8fff8;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #006666 !important; font-weight: 800 !important; font-size: 1.4rem !important;
    }
    [data-baseweb="radio"] [aria-checked="true"] > div:first-child {
        background-color: #006666 !important; border-color: #006666 !important;
    }

    /* TABLEAUX EN VERT + GRAS */
    .dataframe thead th {
        background-color: #006666 !important; color: white !important;
        font-weight: 900 !important; text-align: center !important; font-size: 1.1rem !important;
    }
    .dataframe tbody td {
        font-weight: 700 !important; color: #006666 !important;
        text-align: center !important; background-color: #f8fff8 !important;
    }
    .dataframe tbody td[data-highlight="True"] {
        background-color: #d4edda !important; color: #004d40 !important; font-weight: 900 !important;
    }
    /* Toutes les cellules de tous les tableaux → en gras + vert foncé */
    .dataframe td, .dataframe th {
        font-weight: 800 !important;
        color: #006666 !important;
        text-align: center !important;
        background-color: #f8fff8 !important;
    }
    
    /* En-tête des tableaux → vert foncé + texte blanc */
    .dataframe thead th {
        background-color: #006666 !important;
        color: white !important;
        font-weight: 900 !important;
        font-size: 1.1rem !important;
    }
    
    /* Meilleur score → fond vert clair + texte encore plus foncé */
    .dataframe td.highlighted, .dataframe th.highlighted {
        background-color: #d4edda !important;
        color: #004d40 !important;
        font-weight: 900 !important;
    }        
</style>
""", unsafe_allow_html=True)
# =============================================
# HEADER
# =============================================
st.markdown("""
<div class="medical-header">
    <h1>SVM à Noyaux</h1>
    <h2>Conception, Sélection de Noyaux et Comparaison Empirique<br>
    Application à la Détection de Cancer du Sein</h2>
</div>
""", unsafe_allow_html=True)

# =============================================
# CHARGEMENT
# =============================================
@st.cache_data
def load_results():
    try:
        with open('models/saved_models/all_results.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Fichier all_results.pkl introuvable. Exécutez train_svm.py d'abord.")
        st.stop()

@st.cache_resource
def load_model(dataset, kernel):
    path = f'models/saved_models/{dataset}_{kernel}_model.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

results = load_results()

# =============================================
# NOMS LISIBLES DES FEATURES
# =============================================
FEATURE_NAMES = {
    'Coimbra': [
        "Âge (années)", "IMC (kg/m²)", "Glucose à jeun (mg/dL)", "Insuline (µU/mL)",
        "HOMA-IR", "Leptine (ng/mL)", "Adiponectine (µg/mL)", "Résistine (ng/mL)", "MCP-1 (pg/mL)"
    ],
    'Haberman': [
        "Âge au moment de l’opération", "Année de l’opération (ex: 63 = 1963)", "Nb ganglions positifs"
    ],
    'WDBC': [
        "Rayon moyen", "Texture moyenne", "Périmètre moyen", "Aire moyenne",
        "Lissage moyen", "Compacité moyenne", "Concavité moyenne", "Points concaves moyen",
        "Symétrie moyenne", "Dimension fractale moyenne"
    ],
    'Wisconsin': [
        "Épaisseur du clump", "Uniformité taille cellules", "Uniformité forme cellules",
        "Adhésion marginale", "Taille cellules épithéliales", "Noyaux nus",
        "Chromatine", "Nucléoles", "Mitoses"
    ]
}

DATASET_INFO = {
    'Coimbra':     {'desc': 'Marqueurs métaboliques (116 patientes)', 'target_names': ['Sain', 'Malade']},
    'Haberman':    {'desc': 'Survie post-chirurgie (306 patientes)', 'target_names': ['Survie > 5 ans', 'Décès < 5 ans']},
    'WDBC':        {'desc': 'Caractéristiques cellulaires (569 patientes)', 'target_names': ['Bénin', 'Malin']},
    'Wisconsin':   {'desc': 'Cytologie manuelle (683 patientes)', 'target_names': ['Bénin', 'Malin']}
}

# =============================================
# SIDEBAR EN VERT PARFAIT
# =============================================
with st.sidebar:
    st.markdown("<h2>Navigation</h2>", unsafe_allow_html=True)

    page = st.radio(
        "",
        ["Accueil", "Résultats", "Comparaison", "Meilleur Noyau", "Prédiction"],
        label_visibility="collapsed"
    )

# =============================================
# PAGES (simplifiées pour claires)
# =============================================
if page == "Accueil":
    col1, col2, col3 = st.columns(3)
    col1.metric("Datasets", len(results))
    col2.metric("Modèles", sum(len(v) for v in results.values()))
    col3.metric("Accuracy moyenne", f"{np.mean([v['accuracy'] for d in results.values() for v in d.values()]):.1%}")
    st.success("Tous les modèles sont chargés avec succès")

elif page == "Résultats":
    st.header("Tableau des Performances")
    df = pd.DataFrame([
        {'Dataset': ds, 'Noyau': k.upper(), 'Accuracy': f"{v['accuracy']:.4f}"}
        for ds, ks in results.items() for k, v in ks.items()
    ])
    st.dataframe(df.style.highlight_max(axis=0, color='#d4edda'), use_container_width=True)
elif page == "Comparaison":
    st.header("Comparaison Complète des Performances")

    # Sélection de la métrique
    st.markdown(
        "<p style='color:#006666;" \
        " font-weight:800; " \
        "font-size:1.2rem; " \
        "margin-bottom:0.5rem;'>"
        "<strong>Métrique à visualiser</strong>"
    "</p>", 
    unsafe_allow_html=True
)

    metric = st.selectbox(
    label=" ",  # on met un espace pour cacher le label par défaut
    options=['accuracy', 'f1', 'auc'],
    format_func=lambda x: {
        'accuracy': 'Accuracy',
        'f1': 'F1-Score',
        'auc': 'AUC-ROC (Area Under Curve)'
    }[x],
    index=2,
    label_visibility="collapsed"  # cache le label gris de Streamlit
)
    # Récupération des données
    data = []
    for ds, kernels in results.items():
        for k, v in kernels.items():
            score = v.get('auc' if metric == 'auc' else metric, 0)
            data.append({
                'Dataset': ds,
                'Noyau': k.upper(),
                'Score': score,
                'Score %': score * 100
            })
    df_plot = pd.DataFrame(data)

    # Graphique
    fig = px.bar(
        df_plot,
        x='Dataset', y='Score %', color='Noyau',
        barmode='group',
        text=df_plot['Score %'].apply(lambda x: f"{x:.1f}%"),
        color_discrete_map={'LINEAR':'#006666', 'POLY':'#e67e22', 'RBF':'#2980b9', 'SIGMOID':'#c0392b'},
        title=f"Comparaison des { {'accuracy':'Accuracy','f1':'F1-Score','auc':'AUC-ROC'}[metric] }",
        height=600
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        yaxis=dict(tickformat=".1f", range=[40, 105], title="Performance (%)"),
        legend_title="Noyau SVM",
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tableau récapitulatif des 3 métriques
    st.markdown("### Tableau récapitulatif complet")
    summary = []
    for ds, kernels in results.items():
        row = {'Dataset': ds}
        for k, v in kernels.items():
            row[f"{k.upper()} Acc"] = f"{v['accuracy']:.1%}"
            row[f"{k.upper()} F1"]  = f"{v['f1']:.3f}"
            row[f"{k.upper()} AUC"] = f"{v.get('auc',0):.3f}"
        summary.append(row)
    st.dataframe(
        pd.DataFrame(summary).set_index('Dataset')
        .style.highlight_max(axis=1, color="#9b8585"),
        use_container_width=True
    )
 

elif page == "Meilleur Noyau":
    # Créer deux onglets
    tab1, tab2 = st.tabs([" Meilleur Noyau par Dataset", " Meilleur Noyau Global"])
    
    with tab1:
        st.header("  Meilleur Noyau par Dataset")
        best_kernels = []
        for ds, kernels in results.items():
            best_kernel = max(kernels.items(), key=lambda x: x[1]['accuracy'])
            best_kernels.append({
                'Dataset': ds,
                'Meilleur Noyau': best_kernel[0].upper(),
                'Accuracy': f"{best_kernel[1]['accuracy']:.4f}",
                'F1-Score': f"{best_kernel[1]['f1']:.4f}",
                'AUC-ROC': f"{best_kernel[1].get('auc',0):.4f}"
            })
        st.dataframe(
            pd.DataFrame(best_kernels).set_index('Dataset')
            .style.highlight_max(axis=0, color="#80E79F"),
            use_container_width=True
        )
    
    with tab2:
        st.header("  Meilleur Noyau Global")
        # Meilleur noyau global sur tous les datasets
        overall_best = None
        for ds, kernels in results.items():
            for k, v in kernels.items():
                if overall_best is None or v['accuracy'] > overall_best[2]['accuracy']:
                    overall_best = (ds, k, v)
        
        st.success(f" Meilleur noyau global : **{overall_best[1].upper()}** sur le dataset **{overall_best[0]}** "
                   f"avec une accuracy de **{overall_best[2]['accuracy']:.4f}**")
        
        # Afficher les détails
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{overall_best[2]['accuracy']:.4f}")
        with col2:
            st.metric("F1-Score", f"{overall_best[2]['f1']:.4f}")
        with col3:
            st.metric("AUC-ROC", f"{overall_best[2].get('auc', 0):.4f}")

elif page == "Prédiction":
    # Titre principal
    st.markdown("""
    <div style="background: linear-gradient(135deg, #006666, #008888); color: white; padding: 2.5rem; border-radius: 50px;
                text-align: center; font-size: 2.8rem; font-weight: 900; margin: 2rem 0 3rem 0; box-shadow: 0 15px 40px rgba(0,102,102,0.4);">
        Prédiction sur un Nouveau Cas
    </div>
    """, unsafe_allow_html=True)

    # === CONFIGURATION COMPLÈTE + EXPLICATIONS ===
    CONFIG = {
        'Coimbra': {
            'labels': ["Âge (années)", "IMC (kg/m²)", "Glucose à jeun (mg/dL)", "Insuline (µU/mL)",
                       "HOMA-IR", "Leptine (ng/mL)", "Adiponectine (µg/mL)", "Résistine (ng/mL)", "MCP-1 (pg/mL)"],
            'help': [
                "Âge de la patiente", "Indice de masse corporelle", "Glycémie à jeun", "Insulinémie à jeun",
                "Indice d’insulinorésistance", "Hormone de la satiété", "Hormone anti-inflammatoire", 
                "Hormone pro-inflammatoire", "Chimiokine inflammatoire"
            ],
            'ranges': [(24,89), (18,39), (60,200), (2.5,58), (0.4,25), (4,125), (1.6,38), (3,82), (45,1700)],
            'defaults': [48, 25.0, 90, 10.0, 2.5, 30.0, 12.0, 10.0, 500]
        },
        'Haberman': {
            'labels': ["Âge", "Année de l’opération (ex: 63 = 1963)", "Nombre de ganglions positifs"],
            'help': [
                "Âge au moment de l’opération", "Année de l’intervention chirurgicale", 
                "Nombre de ganglions lymphatiques positifs (0 à 52) → facteur pronostique majeur"
            ],
            'ranges': [(30,83), (58,69), (0,52)],
            'defaults': [52, 63, 3]
        },
        'WDBC': {
            'labels': [
                "Rayon moyen", "Texture moyenne", "Périmètre moyen", "Aire moyenne", "Lissage moyen",
                "Compacité moyenne", "Concavité moyenne", "Points concaves moyen", "Symétrie moyenne", "Fractale moyenne",
                "Rayon erreur std", "Texture erreur std", "Périmètre erreur std", "Aire erreur std", "Lissage erreur std",
                "Compacité erreur std", "Concavité erreur std", "Points concaves erreur std", "Symétrie erreur std", "Fractale erreur std",
                "Rayon pire", "Texture pire", "Périmètre pire", "Aire pire", "Lissage pire",
                "Compacité pire", "Concavité pire", "Points concaves pire", "Symétrie pire", "Fractale pire"
            ],
            'help': [
                "Rayon moyen des noyaux cellulaires", "Texture moyenne (écart-type gris)", "Périmètre moyen", "Aire moyenne",
                "Variation locale des longueurs de rayon", "Compacité = (périmètre² / aire) – 1", "Gravité des portions concaves",
                "Nombre de portions concaves", "Symétrie des noyaux", "Dimension fractale moyenne",
                "Écart-type du rayon", "Écart-type de la texture", "Écart-type du périmètre", "Écart-type de l'aire",
                "Écart-type du lissage", "Écart-type de la compacité", "Écart-type de la concavité", "Écart-type des points concaves",
                "Écart-type de la symétrie", "Écart-type fractale",
                "Valeur maximale du rayon (pire noyau)", "Texture maximale", "Périmètre maximal", "Aire maximale",
                "Lissage maximal", "Compacité maximale", "Concavité maximale", "Points concaves maximum", "Symétrie maximale", "Fractale maximale"
            ],
            'ranges': [(6,29), (9,40), (43,188), (143,2501), (0.05,0.16), (0.02,0.35), (0,0.43), (0,0.2), (0.1,0.32), (0.05,0.1),
                       (0.1,2.8), (0.3,6.1), (0.7,21.9), (6.8,542), (0.001,0.03), (0.002,0.13), (0,0.4), (0,0.05), (0.008,0.08), (0.0001,0.006),
                       (7.9,36), (12,49), (50,251), (185,4254), (0.07,0.19), (0.03,1.0), (0,1.1), (0,0.29), (0.16,0.66), (0.055,0.21)],
            'defaults': [14.13, 20.99, 91.86, 609.2, 0.091, 0.106, 0.086, 0.045, 0.181, 0.057,
                         0.569, 1.159, 2.42, 157.0, 0.007, 0.020, 0.027, 0.010, 0.018, 0.002,
                         15.78, 24.15, 101.4, 686.9, 0.096, 0.134, 0.108, 0.061, 0.208, 0.065]
        },
        'Wisconsin': {
            'labels': ["Épaisseur du clump", "Uniformité taille cellules", "Uniformité forme cellules",
                       "Adhésion marginale", "Taille cellules épithéliales", "Noyaux nus",
                       "Chromatine", "Nucléoles", "Mitoses"],
            'help': [
                "Épaisseur du groupe cellulaire", "Uniformité de la taille des cellules", "Uniformité de la forme",
                "Adhésion des cellules au bord", "Taille moyenne des cellules épithéliales", "Nombre de noyaux sans cytoplasme",
                "Texture de la chromatine", "Nombre de nucléoles visibles", "Nombre de mitoses par champ"
            ],
            'ranges': [(1,10)] * 9,
            'defaults': [5, 4, 4, 4, 4, 4, 4, 4, 1]
        }
    }

    # === SÉLECTION + NOYAU ===
    dataset = st.selectbox("Choisir le type d'analyse", list(results.keys()),
                           format_func=lambda x: f"{x} — {DATASET_INFO[x]['desc']}")
    
    best_kernel = max(results[dataset].items(), key=lambda x: x[1]['accuracy'])[0]
    kernel = st.radio("Noyau SVM", ['linear','poly','rbf','sigmoid'],
                      format_func=str.upper, index=['linear','poly','rbf','sigmoid'].index(best_kernel))

    st.info(DATASET_INFO[dataset]['desc'])

    # === BOUTON AIDE – EXPLICATIONS COMPLÈTES ===
    if st.button("Aide – Comprendre les variables", use_container_width=True):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #006666, #008888); color: white; padding: 2.2rem; border-radius: 50px;
                    text-align: center; font-size: 2.4rem; font-weight: 900; margin: 2rem 0;">
            Signification des Variables
        </div>
        """, unsafe_allow_html=True)

        for ds_name, cfg in CONFIG.items():
            with st.expander(f"{ds_name} – {DATASET_INFO[ds_name]['desc']}", expanded=(ds_name == dataset)):
                for label, expl in zip(cfg['labels'], cfg['help']):
                    st.markdown(f"- **{label}** : {expl}")

    # === TITRE ENTRÉE DES VALEURS ===
    st.markdown("""
    <div style="background: linear-gradient(135deg, #006666, #008888); color: white; padding: 1.8rem 3rem; border-radius: 50px;
                text-align: center; font-size: 2rem; font-weight: 800; margin: 3rem 0 2rem 0; box-shadow: 0 10px 30px rgba(0,102,102,0.3);">
        Entrer les valeurs du patient
    </div>
    """, unsafe_allow_html=True)

    # === FORMULAIRE ===
    config = CONFIG[dataset]
    values = []
    valid = True

    cols = st.columns(3)
    for i, (label, (min_v, max_v), default_v, expl) in enumerate(zip(
        config['labels'], config['ranges'], config['defaults'], config['help']
    )):
        with cols[i % 3]:
            if dataset == 'Wisconsin':
                val = st.slider(label, min_v, max_v, default_v, help=expl, key=f"val_{i}")
            else:
                val = st.number_input(
                    label,
                    min_value=float(min_v), max_value=float(max_v),
                    value=float(default_v),
                    step=0.1 if max_v < 100 else 1.0,
                    help=expl,
                    key=f"val_{i}"
                )
            values.append(val)
            if val < min_v or val > max_v:
                st.error(f"{label} : hors plage ({min_v} – {max_v})")
                valid = False

    # Pour WDBC : on complète à 30 features si besoin
    if dataset == 'WDBC' and len(values) < 30:
        values += [config['defaults'][0]] * (30 - len(values))

    st.markdown("---")

    # === ANALYSER ===
    if st.button("ANALYSER LE CAS", type="primary", use_container_width=True):
        if not valid:
            st.error("Corrigez les valeurs en rouge avant de continuer.")
        else:
            model = load_model(dataset, kernel.lower())
            if model is None:
                st.error("Modèle introuvable.")
            else:
                X = np.array(values[:30] if dataset=='WDBC' else values, dtype=float).reshape(1, -1)
                pred = int(model.predict(X)[0])
                proba = model.predict_proba(X)[0]
                classe = DATASET_INFO[dataset]['target_names'][pred]
                confiance = proba[pred] * 100

                st.markdown(f"<div class='prediction-box {'benin' if pred==0 else 'malin'}'>RÉSULTAT : {classe.upper()}</div>", 
                            unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                c1.metric("Diagnostic", classe)
                c1.metric("Confiance", f"{confiance:.1f}%")
                c2.plotly_chart(go.Figure(go.Indicator(
                    mode="gauge+number", value=confiance,
                    title="Confiance (%)", number={'suffix': "%"}
                )), use_container_width=True)

                st.success("Analyse réussie")
                st.warning("À but éducatif uniquement — Consultez un médecin")

# =============================================
# FOOTER
# =============================================
st.markdown("""
<div style='text-align:center; padding:3rem 0 2rem;'>
    <h4 style='color:#006666; font-weight:700; margin:0;'>
        Détection du cancer du sein — SVM à Noyaux 
    </h4>
</div>

""", unsafe_allow_html=True)


