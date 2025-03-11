import streamlit as st
import pandas as pd
import pickle
import shap
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
#https://projet8-ys94.onrender.com/
# === 1) CHARGEMENT DU MODÈLE ET DES DONNÉES ===
model = pickle.load(open('model.pkl', 'rb'))
data = pd.read_csv('test_df_api.csv')
data_train = pd.read_csv('train_df_api.csv')

# Prétraitement / scaling
data_scaled = data.copy()
data_train_scaled = data_train.copy()

# Création de l'explainer SHAP (modèle type arbre)
explainer = shap.TreeExplainer(model['classifier'])

# === 2) SIDEBAR POUR CHOISIR LE CLIENT ===
st.sidebar.header("🔍 Sélection du client")
st.sidebar.header("Client ID Selection")
client_id = st.sidebar.selectbox("Choose a Client ID:", data['SK_ID_CURR'])

# === 3) TITRE ET INTRODUCTION ===

st.title("📊 Dashboard Crédit Accessible")
st.write("Ce dashboard affiche la prédiction et l'explication SHAP pour un client donné.")

# === 4) VÉRIFICATION DE L'EXISTENCE DU CLIENT ===
if client_id:
    if client_id not in list(data['SK_ID_CURR']):
        st.error("Client ID not found in the database.")
    else:
        # === 4.1) INFORMATIONS DU CLIENT ===
        st.subheader("Client Information")
        client_data = data[data['SK_ID_CURR'] == client_id]
        st.write(client_data)

        # === 4.2) PRÉDICTION DU RISQUE DE DÉFAUT ===
        st.subheader("📈 Probabilité de défaut")
        info_client = client_data.drop('SK_ID_CURR', axis=1)
        prediction = model.predict_proba(info_client)[0][1]
        st.write(f"Default Probability: {prediction:.3f}")

        # Décision selon un seuil
        threshold = 0.5
        decision = "✅ Approuvé" if prediction < threshold else "❌ Refusé"
       
        st.markdown(f"### {decision}")

        # === Création de la jauge ===
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Default Probability"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 0.5], 'color': "blue"},
                    {'range': [0.5, 1], 'color': "orange"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction
                }
            }
        ))
        st.plotly_chart(fig_gauge)
        # === 5) EXPLICATION SHAP GLOBALE ===
        st.subheader("🔬 Explication SHAP Globale")
        # Calcul des SHAP values globales
        shap_vals_global = explainer.shap_values(data_scaled.drop('SK_ID_CURR', axis=1))

        # Graphique en barres (summary_plot global)
        fig_global, ax_global = plt.subplots()
        shap.summary_plot(
            shap_vals_global,
            data_scaled.drop('SK_ID_CURR', axis=1),
            plot_type='bar',
            show=False
        )
        st.pyplot(fig_global)

        # === 5) EXPLICATION SHAP GLOBALE ===
        st.subheader("SHAP Global Explanation")
        shap_vals_global = explainer.shap_values(data_scaled.drop('SK_ID_CURR', axis=1))
        if isinstance(shap_vals_global, list) and len(shap_vals_global) == 2:
            shap_vals_global_class1 = shap_vals_global[1]
        else:
            shap_vals_global_class1 = shap_vals_global

        X_global = data_scaled.drop('SK_ID_CURR', axis=1)
        shap_values_exp = shap.Explanation(
            values=shap_vals_global_class1,
            data=X_global,
            feature_names=X_global.columns
        )

        fig_global, ax_global = plt.subplots()
        shap.plots.beeswarm(shap_values_exp, max_display=20, show=False)
        st.pyplot(fig_global)

        # === 6) EXPLICATION SHAP LOCALE ===
        st.subheader("🔍 Explication SHAP Locale")
        X_client = client_data.drop('SK_ID_CURR', axis=1)
        shap_values_local = explainer(X_client)
        if isinstance(shap_values_local, list):
            local_explanation = shap_values_local[1][0]
        else:
            local_explanation = shap_values_local[0]

        fig_local = plt.figure()
        shap.waterfall_plot(local_explanation, show=False)
        st.pyplot(fig_local)

       
        # === 8) ANALYSE DES VOISINS ===
        st.subheader("🔍 Analyse des voisins similaires")
        def get_data_voisins(client_id: int):
            features = list(data_train_scaled.columns)
            features.remove('SK_ID_CURR')
            features.remove('TARGET')

            nn = NearestNeighbors(n_neighbors=10, metric='euclidean')
            nn.fit(data_train_scaled[features])
            reference_observation = data_scaled[data_scaled['SK_ID_CURR'] == client_id][features].values
            indices = nn.kneighbors(reference_observation, return_distance=False)
            df_voisins = data_train.iloc[indices[0], :]
            return df_voisins.to_json()

        df_voisins = get_data_voisins(client_id)
        if df_voisins:
            df_voisins_df = pd.read_json(df_voisins)
            st.subheader("Voisins Similaires")
            st.write(df_voisins_df)

            feature_options = df_voisins_df.columns.tolist()
            selected_feature = st.selectbox("Choisissez une feature :", feature_options)

            def distribution(feature, id_client, df):
                fig, ax = plt.subplots(figsize=(15, 10))
                ax.hist(df[df['TARGET'] == 0][feature], bins=30, label='Accordé')
                ax.hist(df[df['TARGET'] == 1][feature], bins=30, label='Refusé')
                observation_value = data.loc[data['SK_ID_CURR'] == id_client][feature].values
                ax.axvline(observation_value, color='green', linestyle='dashed', linewidth=2, label='Client')
                ax.legend()
                st.pyplot(fig)

            distribution(selected_feature, client_id, df_voisins_df)

            selected_feature_x = st.selectbox("Choisissez une feature pour l'axe X :", feature_options)
            selected_feature_y = st.selectbox("Choisissez une feature pour l'axe Y :", feature_options)

            def scatter(id_client, feature_x, feature_y, df):
                fig, ax = plt.subplots(figsize=(10, 6))
                data_accord = df[df['TARGET'] == 0]
                data_refus = df[df['TARGET'] == 1]
                ax.scatter(data_accord[feature_x], data_accord[feature_y], color='blue', alpha=0.5, label='Accordé')
                ax.scatter(data_refus[feature_x], data_refus[feature_y], color='red', alpha=0.5, label='Refusé')
                observation_x = data.loc[data['SK_ID_CURR'] == id_client][feature_x].values[0]
                observation_y = data.loc[data['SK_ID_CURR'] == id_client][feature_y].values[0]
                ax.scatter(observation_x, observation_y, marker='*', s=200, color='black', label='Client')
                ax.legend()
                st.pyplot(fig)

            scatter(client_id, selected_feature_x, selected_feature_y, df_voisins_df)
        else:
            st.error("Aucun voisin trouvé.")

        # === 6) MODIFICATION DES FEATURES ET NOUVELLE PRÉDICTION ===
        st.subheader("✏️ Modification des Features et Nouvelle Prédiction")

        # Extraire les 10 features les plus influentes
        shap_importance = np.abs(local_explanation.values)
        top_10_features = np.argsort(shap_importance)[-10:]
        top_10_feature_names = [X_client.columns[i] for i in top_10_features]

        # Créer une copie complète des features du client
        modifiable_client_data = client_data.drop('SK_ID_CURR', axis=1).copy()
        modified_features = {}

        st.write("Seules les 10 features les plus influentes sont modifiables :")
        for feature in top_10_feature_names:
            if modifiable_client_data[feature].dtype in [np.float64, np.int64]:
                min_val = float(data_train[feature].min())
                max_val = float(data_train[feature].max())
                default_val = float(modifiable_client_data[feature].values[0])

                if min_val == max_val:
                    min_val -= 1
                    max_val += 1

                modified_features[feature] = st.slider(
                    f"Modifier {feature} :",
                    min_val, max_val, default_val
                )
            else:
                unique_values = data_train[feature].unique()
                default_val = modifiable_client_data[feature].values[0]
                modified_features[feature] = st.selectbox(
                    f"Modifier {feature} :",
                    options=unique_values,
                    index=list(unique_values).index(default_val)
                )

        # Appliquer les modifications uniquement aux 10 features sélectionnées
        for feature, value in modified_features.items():
            modifiable_client_data[feature] = value

        # Vérification du nombre de features
        st.write(f"Nombre de features avant prédiction : {modifiable_client_data.shape[1]}")

        # Transformation correcte du DataFrame
        new_client_data = modifiable_client_data.copy().astype(float)

        # Vérifier si toutes les features attendues sont bien présentes (exclure SK_ID_CURR et TARGET)
        expected_features = set(data_train.columns) - {"SK_ID_CURR", "TARGET"}
        missing_features = expected_features - set(new_client_data.columns)

        if missing_features:
            st.error(f"Attention : il manque des features dans les données d'entrée du modèle : {missing_features}")
        else:
            # Nouvelle prédiction
            new_prediction = model.predict_proba(new_client_data)[0][1]
            st.write(f"Nouvelle probabilité de défaut : {new_prediction:.3f}")

            new_decision = "✅ Approuvé" if new_prediction < threshold else "❌ Refusé"
            st.markdown(f"### {new_decision}")

            # Nouvelle explication SHAP locale
            new_shap_values_local = explainer(new_client_data)
            new_local_explanation = new_shap_values_local[1][0] if isinstance(new_shap_values_local, list) else new_shap_values_local[0]

            fig_new_local = plt.figure()
            shap.waterfall_plot(new_local_explanation, show=False)
            st.pyplot(fig_new_local)

