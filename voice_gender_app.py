import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

df = pd.read_csv("vocal_gender_features_new.csv")

X_full = df.drop(columns=["label"])
y = df["label"]

# Train-test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X_full)

svm_model = SVC(probability=True, C=10, kernel='rbf', random_state=42)
svm_model.fit(X_scaled_full, y)
best_models = {"SVM": svm_model}

st.set_page_config(page_title="Human Voice Classification and Clustering", layout="wide")
st.sidebar.title("Pages")
page = st.sidebar.radio("Go to", ["Introduction", "EDA", "Prediction", "Best Classification", "Best Clustering"])

# Introduction
if page == "Introduction":
    st.title("Human Voice Classification and Clustering")
    st.markdown("""
    ## About the Project
    
    Human voice carries rich information not only about speech content but also about the speaker’s identity, gender, age, and emotion. 
    This project explores voice-based gender classification and clustering using machine learning techniques.

    We use audio signal features — such as **spectral centroid**, **zero-crossing rate**, **MFCCs**, and **pitch-related descriptors** — 
    to build models that can:
    
    - Automatically **classify voice samples by gender**
    - **Cluster similar voices** based on their acoustic features

    This approach has real-world applications in:
    - Voice assistants and smart devices
    - Speaker profiling and security systems
    - Personalized media and accessibility tools

    The dataset consists of pre-extracted numerical features from voice recordings labeled by gender. 
    Machine learning models like Random Forest, SVM, and Neural Networks are used for classification, while clustering is done via KMeans, DBSCAN, and Hierarchical methods.
    """)

# EDA
elif page == "EDA":
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("Analysis of acoustic voice features grouped by gender")

    eda_questions = {
        "1. Are there significant differences in spectral centroid between the two genders?": 
            lambda ax: sns.boxplot(data=df, x='label', y='mean_spectral_centroid', ax=ax),

        "2. Is there a strong relationship between spectral bandwidth and pitch variability?": 
            lambda ax: sns.scatterplot(data=df, x='mean_spectral_bandwidth', y='std_pitch', hue='label', ax=ax),

        "3. Do male and female voices differ in mean pitch?": 
            lambda ax: sns.histplot(data=df, x='mean_pitch', hue='label', kde=True, ax=ax),

        "4. Is spectral contrast higher in male or female voices?": 
            lambda ax: sns.violinplot(data=df, x='label', y='mean_spectral_contrast', ax=ax),

        "5. How does zero-crossing rate vary across genders?": 
            lambda ax: sns.boxplot(data=df, x='label', y='zero_crossing_rate', ax=ax),

        "6. Is pitch standard deviation (pitch variability) higher for one gender?": 
            lambda ax: sns.histplot(data=df, x='std_pitch', hue='label', kde=True, ax=ax),

        "7. What is the correlation between mean_pitch and spectral_rolloff?": 
            lambda ax: sns.scatterplot(data=df, x='mean_pitch', y='mean_spectral_rolloff', hue='label', ax=ax),

        "8. Does spectral skew or kurtosis reveal voice timbre differences?": 
            "multi_plot", 

        "9. Are there clusters of voices based on spectral_centroid and rms_energy?": 
            lambda ax: sns.scatterplot(data=df, x='mean_spectral_centroid', y='rms_energy', hue='label', ax=ax),

        "10. Is there redundancy among spectral features (e.g., centroid vs rolloff)?": 
            lambda ax: sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
    }

    selected_q = st.selectbox("Select EDA Question", list(eda_questions.keys()))
    st.subheader(selected_q)

    if eda_questions[selected_q] == "multi_plot":
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.boxplot(data=df, x='label', y='spectral_kurtosis', ax=axes[0])
        axes[0].set_title("Spectral Kurtosis by Gender")
        sns.boxplot(data=df, x='label', y='spectral_skew', ax=axes[1])
        axes[1].set_title("Spectral Skew by Gender")
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_func = eda_questions[selected_q]
        plot_func(ax)
        st.pyplot(fig)

# Prediction
elif page == "Prediction":
    st.header("Prediction Using Top 15 Features")

    # Feature selection applied here only
    selector = SelectKBest(score_func=f_classif, k=15)
    X_selected = selector.fit_transform(X_full, y)
    selected_features = X_full.columns[selector.get_support()].tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    svm = SVC(probability=True, C=10, kernel='rbf', random_state=42)
    svm.fit(X_scaled, y)
    best_models["SVM"] = svm

    input_data = []
    st.markdown("### Input Feature Values")
    for feature in selected_features:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        val = st.slider(
            f"{feature}",
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=(max_val - min_val) / 100,
        )
        input_data.append(val)

    if st.button("Predict Gender"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        pred = svm.predict(input_scaled)
        proba = svm.predict_proba(input_scaled)[0]

        predicted_gender = "Male" if pred[0] == 1 else "Female"
        st.success(f"Predicted Gender: {predicted_gender}")
        st.info(f"Confidence: Male: {proba[1]*100:.2f}%, Female: {proba[0]*100:.2f}%")

        st.subheader("Where does this voice lie in the cluster space?")
        agg = AgglomerativeClustering(n_clusters=2)
        cluster_labels = agg.fit_predict(X_scaled)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        input_pca = pca.transform(input_scaled)

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis", s=40, alpha=0.6, label="Dataset")
        ax.scatter(input_pca[0, 0], input_pca[0, 1], color="red", s=100, marker="X", label="Your Voice")
        ax.legend()
        ax.set_title("Cluster Position of Your Input Voice (PCA Reduced)")
        st.pyplot(fig)

# Best Classification
elif page == "Best Classification":
    st.header("Best Classification Model: SVM")

    y_pred = best_models["SVM"].predict(X_scaled_full)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    st.write("### Metrics:")
    st.write(f"Accuracy: {acc:.4f}")
    st.write(f"Precision: {prec:.4f}")
    st.write(f"Recall: {rec:.4f}")
    st.write(f"F1 Score: {f1:.4f}")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Female", "Male"])
    disp.plot(ax=ax, cmap='Blues')
    st.pyplot(fig)

# Best Clustering
elif page == "Best Clustering":
    st.header("Best Clustering Model: Agglomerative Clustering")

    agg_model = AgglomerativeClustering(n_clusters=2)
    cluster_labels = agg_model.fit_predict(X_scaled_full)

    silhouette = silhouette_score(X_scaled_full, cluster_labels)

    st.write(f"### Silhouette Score: {silhouette:.3f}")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled_full)

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=15)
    legend_labels = [f"Cluster {i}" for i in np.unique(cluster_labels)]
    legend = ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Clusters")
    ax.set_title("Agglomerative Clustering (PCA Projection)")
    st.pyplot(fig)
