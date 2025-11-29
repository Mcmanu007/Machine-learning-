import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.graph_objects as go
import plotly.express as px
import io
import requests

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Clustering Models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.mixture import GaussianMixture

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression as LR_NLP

# Custom CSS for styling
st.set_page_config(page_title="AI/ML Studio", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }
    
    /* Card-like containers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px 20px;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.3);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #fff;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #ff6b6b 0%, #ff8e53 100%);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 30px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    /* DataFrames */
    .dataframe {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
    }
    
    /* Select boxes */
    .stSelectbox, .stMultiSelect {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        border: 2px dashed rgba(255, 255, 255, 0.5);
        padding: 20px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = []
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'text_column' not in st.session_state:
    st.session_state.text_column = None

# Header
st.markdown("<h1 style='text-align: center; font-size: 3.5em;'>🤖 AI/ML Studio Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 1.2em;'>Your Complete No-Code Machine Learning Platform</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation with icons
st.sidebar.markdown("## 🎯 Navigation")
page = st.sidebar.radio(
    "",
    ["🏠 Home", "📊 Data Import", "🎯 Supervised Learning", "🔍 Clustering", "📝 NLP Tasks", "📈 Model Comparison", "🧪 Test & Deploy"],
    label_visibility="collapsed"
)

# Home Page
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.3);'>
            <h2 style='color: white;'>🎯 Supervised Learning</h2>
            <p style='color: white;'>Classification & Regression models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 30px; border-radius: 15px; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.3);'>
            <h2 style='color: white;'>🔍 Clustering</h2>
            <p style='color: white;'>Unsupervised pattern discovery</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 30px; border-radius: 15px; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.3);'>
            <h2 style='color: white;'>📝 NLP</h2>
            <p style='color: white;'>Text analytics & classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: rgba(255, 255, 255, 0.95); padding: 30px; border-radius: 15px; box-shadow: 0 8px 16px rgba(0,0,0,0.3);'>
        <h2 style='color: #667eea;'>✨ Features</h2>
        <ul style='font-size: 1.1em; color: #333;'>
            <li>📥 <b>Import from Kaggle</b> - Direct dataset integration</li>
            <li>🤖 <b>10+ ML Algorithms</b> - Classification, Regression, Clustering</li>
            <li>📝 <b>NLP Pipeline</b> - Text classification, sentiment analysis, topic modeling</li>
            <li>📊 <b>Interactive Visualizations</b> - Compare model performance</li>
            <li>🧪 <b>Real-time Testing</b> - Test your models instantly</li>
            <li>🎨 <b>Auto Data Preprocessing</b> - Handle missing values, encoding, scaling</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.info("👈 **Get Started:** Use the sidebar to navigate through different sections!")

# Data Import Page
elif page == "📊 Data Import":
    st.markdown("<h2>📊 Import Your Dataset</h2>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📁 Upload CSV", "🏆 Import from Kaggle"])
    
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("✅ File uploaded successfully!")
            
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("### 📊 Dataset Statistics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📏 Rows", df.shape[0])
            col2.metric("📐 Columns", df.shape[1])
            col3.metric("❌ Missing Values", df.isnull().sum().sum())
            col4.metric("💾 Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🏆 Import Dataset from Kaggle")
        
        st.info("📝 **Instructions:** Enter the Kaggle dataset path (e.g., 'username/dataset-name') or direct CSV URL")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            kaggle_path = st.text_input("Kaggle Dataset Path or CSV URL", placeholder="e.g., uciml/iris or https://example.com/data.csv")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            import_btn = st.button("🔽 Import", use_container_width=True)
        
        if import_btn and kaggle_path:
            try:
                with st.spinner("Importing dataset..."):
                    # Try to load from URL
                    if kaggle_path.startswith('http'):
                        df = pd.read_csv(kaggle_path)
                        st.session_state.df = df
                        st.success("✅ Dataset imported successfully from URL!")
                    else:
                        # For Kaggle, provide instructions
                        st.warning("⚠️ For Kaggle imports, please use Kaggle API or download the CSV directly. For now, you can:")
                        st.markdown("""
                        1. Download the dataset from Kaggle
                        2. Use the 'Upload CSV' tab to upload it
                        
                        Or provide a direct CSV URL from Kaggle or other sources.
                        """)
                        
                        # Try a demo dataset
                        st.info("📊 Loading a demo dataset (Iris) for demonstration...")
                        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
                        df = pd.read_csv(url)
                        st.session_state.df = df
                        st.success("✅ Demo dataset loaded!")
                    
                    if st.session_state.df is not None:
                        st.dataframe(st.session_state.df.head(10), use_container_width=True)
                        
            except Exception as e:
                st.error(f"❌ Error importing dataset: {str(e)}")

# Supervised Learning Page
elif page == "🎯 Supervised Learning":
    st.markdown("<h2>🎯 Supervised Learning</h2>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please import your dataset first!")
    else:
        df = st.session_state.df
        
        col1, col2 = st.columns([2, 1])
        with col1:
            problem_type = st.selectbox("🎲 Problem Type", ["Classification", "Regression"])
            st.session_state.problem_type = problem_type
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            
        target_column = st.selectbox("🎯 Target Column", df.columns)
        feature_columns = st.multiselect(
            "📊 Feature Columns",
            [col for col in df.columns if col != target_column],
            default=[col for col in df.columns if col != target_column]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("📊 Test Set Size (%)", 10, 50, 20) / 100
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Prepare Data & Train"):
            try:
                with st.spinner("Preparing data..."):
                    X = df[feature_columns].copy()
                    y = df[target_column].copy()
                    
                    # Handle missing values
                    X = X.fillna(X.mean(numeric_only=True))
                    
                    # Encode categorical variables
                    for col in X.columns:
                        if X[col].dtype == 'object':
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
                    
                    # Encode target if classification
                    if problem_type == "Classification" and y.dtype == 'object':
                        le_target = LabelEncoder()
                        y = le_target.fit_transform(y.astype(str))
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Store in session state
                    st.session_state.X_train = X_train_scaled
                    st.session_state.X_test = X_test_scaled
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.feature_names = feature_columns
                    st.session_state.scaler = scaler
                    
                    st.success("✅ Data prepared successfully!")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("🎓 Training Samples", len(X_train))
                    col2.metric("🧪 Testing Samples", len(X_test))
                
                # Model selection
                st.markdown("### 🤖 Select and Train Models")
                
                if problem_type == "Classification":
                    models_dict = {
                        "Logistic Regression": LogisticRegression(max_iter=1000),
                        "Decision Tree": DecisionTreeClassifier(),
                        "Random Forest": RandomForestClassifier(),
                        "Gradient Boosting": GradientBoostingClassifier(),
                        "AdaBoost": AdaBoostClassifier(),
                        "Support Vector Machine": SVC(),
                        "Naive Bayes": GaussianNB(),
                        "K-Nearest Neighbors": KNeighborsClassifier()
                    }
                else:
                    models_dict = {
                        "Linear Regression": LinearRegression(),
                        "Ridge Regression": Ridge(),
                        "Lasso Regression": Lasso(),
                        "ElasticNet": ElasticNet(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Random Forest": RandomForestRegressor(),
                        "Gradient Boosting": GradientBoostingRegressor(),
                        "AdaBoost": AdaBoostRegressor(),
                        "Support Vector Machine": SVR(),
                        "K-Nearest Neighbors": KNeighborsRegressor()
                    }
                
                selected_models = st.multiselect("Select Models to Train", list(models_dict.keys()))
                
                if st.button("🎯 Train Selected Models"):
                    progress_bar = st.progress(0)
                    for idx, model_name in enumerate(selected_models):
                        with st.spinner(f"Training {model_name}..."):
                            model = models_dict[model_name]
                            model.fit(st.session_state.X_train, st.session_state.y_train)
                            
                            y_train_pred = model.predict(st.session_state.X_train)
                            y_test_pred = model.predict(st.session_state.X_test)
                            
                            if problem_type == "Classification":
                                train_acc = accuracy_score(st.session_state.y_train, y_train_pred)
                                test_acc = accuracy_score(st.session_state.y_test, y_test_pred)
                                precision = precision_score(st.session_state.y_test, y_test_pred, average='weighted', zero_division=0)
                                recall = recall_score(st.session_state.y_test, y_test_pred, average='weighted', zero_division=0)
                                f1 = f1_score(st.session_state.y_test, y_test_pred, average='weighted', zero_division=0)
                                
                                metrics = {
                                    "Train Accuracy": train_acc,
                                    "Test Accuracy": test_acc,
                                    "Precision": precision,
                                    "Recall": recall,
                                    "F1 Score": f1
                                }
                            else:
                                train_r2 = r2_score(st.session_state.y_train, y_train_pred)
                                test_r2 = r2_score(st.session_state.y_test, y_test_pred)
                                mse = mean_squared_error(st.session_state.y_test, y_test_pred)
                                mae = mean_absolute_error(st.session_state.y_test, y_test_pred)
                                rmse = np.sqrt(mse)
                                
                                metrics = {
                                    "Train R² Score": train_r2,
                                    "Test R² Score": test_r2,
                                    "MSE": mse,
                                    "RMSE": rmse,
                                    "MAE": mae
                                }
                            
                            model_info = {
                                "name": model_name,
                                "model": model,
                                "metrics": metrics,
                                "type": "supervised"
                            }
                            st.session_state.trained_models.append(model_info)
                            progress_bar.progress((idx + 1) / len(selected_models))
                    
                    st.success(f"✅ Successfully trained {len(selected_models)} models!")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Clustering Page
elif page == "🔍 Clustering":
    st.markdown("<h2>🔍 Clustering Analysis</h2>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please import your dataset first!")
    else:
        df = st.session_state.df
        
        st.info("ℹ️ Clustering is unsupervised learning - no target column needed!")
        
        feature_columns = st.multiselect(
            "📊 Select Features for Clustering",
            df.columns,
            default=list(df.columns)[:min(5, len(df.columns))]
        )
        
        if feature_columns:
            X = df[feature_columns].copy()
            
            # Preprocessing
            X = X.fillna(X.mean(numeric_only=True))
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            st.markdown("### 🎯 Choose Clustering Algorithm")
            
            col1, col2 = st.columns(2)
            with col1:
                clustering_algo = st.selectbox(
                    "Algorithm",
                    ["K-Means", "DBSCAN", "Agglomerative Clustering", "Gaussian Mixture", "Mean Shift"]
                )
            
            with col2:
                if clustering_algo in ["K-Means", "Agglomerative Clustering", "Gaussian Mixture"]:
                    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            
            if st.button("🚀 Perform Clustering"):
                try:
                    with st.spinner(f"Running {clustering_algo}..."):
                        if clustering_algo == "K-Means":
                            model = KMeans(n_clusters=n_clusters, random_state=42)
                        elif clustering_algo == "DBSCAN":
                            model = DBSCAN(eps=0.5, min_samples=5)
                        elif clustering_algo == "Agglomerative Clustering":
                            model = AgglomerativeClustering(n_clusters=n_clusters)
                        elif clustering_algo == "Gaussian Mixture":
                            model = GaussianMixture(n_components=n_clusters, random_state=42)
                        elif clustering_algo == "Mean Shift":
                            model = MeanShift()
                        
                        clusters = model.fit_predict(X_scaled)
                        
                        # Calculate metrics
                        try:
                            silhouette = silhouette_score(X_scaled, clusters)
                            davies_bouldin = davies_bouldin_score(X_scaled, clusters)
                            calinski = calinski_harabasz_score(X_scaled, clusters)
                            
                            metrics = {
                                "Silhouette Score": silhouette,
                                "Davies-Bouldin Index": davies_bouldin,
                                "Calinski-Harabasz Score": calinski,
                                "Number of Clusters": len(np.unique(clusters))
                            }
                        except:
                            metrics = {"Number of Clusters": len(np.unique(clusters))}
                        
                        st.success("✅ Clustering completed successfully!")
                        
                        # Display metrics
                        cols = st.columns(len(metrics))
                        for i, (metric_name, metric_value) in enumerate(metrics.items()):
                            cols[i].metric(metric_name, f"{metric_value:.4f}" if isinstance(metric_value, float) else metric_value)
                        
                        # Visualization
                        st.markdown("### 📊 Cluster Visualization")
                        
                        # PCA for visualization
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        fig = px.scatter(
                            x=X_pca[:, 0],
                            y=X_pca[:, 1],
                            color=clusters.astype(str),
                            title="Clusters Visualization (PCA)",
                            labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Store model
                        model_info = {
                            "name": clustering_algo,
                            "model": model,
                            "metrics": metrics,
                            "type": "clustering"
                        }
                        st.session_state.trained_models.append(model_info)
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# NLP Tasks Page
elif page == "📝 NLP Tasks":
    st.markdown("<h2>📝 Natural Language Processing</h2>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please import your dataset first!")
    else:
        df = st.session_state.df
        
        nlp_task = st.selectbox(
            "🎯 Select NLP Task",
            ["Text Classification", "Sentiment Analysis", "Topic Modeling"]
        )
        
        text_column = st.selectbox("📄 Select Text Column", df.columns)
        
        if nlp_task in ["Text Classification", "Sentiment Analysis"]:
            target_column = st.selectbox("🎯 Select Target/Label Column", [col for col in df.columns if col != text_column])
        
        col1, col2 = st.columns(2)
        with col1:
            vectorization = st.selectbox("🔤 Vectorization Method", ["TF-IDF", "Count Vectorizer"])
        with col2:
            max_features = st.slider("Max Features", 100, 5000, 1000)
        
        if st.button("🚀 Process & Train"):
            try:
                with st.spinner("Processing text data..."):
                    # Vectorization
                    if vectorization == "TF-IDF":
                        vectorizer = TfidfVectorizer(max_features=max_features)
                    else:
                        vectorizer = CountVectorizer(max_features=max_features)
                    
                    X = vectorizer.fit_transform(df[text_column].fillna(''))
                    st.session_state.vectorizer = vectorizer
                    st.session_state.text_column = text_column
                    
                    if nlp_task in ["Text Classification", "Sentiment Analysis"]:
                        y = df[target_column]
                        if y.dtype == 'object':
                            le = LabelEncoder()
                            y = le.fit_transform(y.astype(str))
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        st.success("✅ Data processed successfully!")
                        
                        # Train models
                        st.markdown("### 🤖 Training NLP Models")
                        
                        models = {
                            "Naive Bayes": MultinomialNB(),
                            "Logistic Regression": LR_NLP(max_iter=1000)
                        }
                        
                        for model_name, model in models.items():
                            with st.spinner(f"Training {model_name}..."):
                                model.fit(X_train, y_train)
                                
                                y_pred = model.predict(X_test)
                                
                                accuracy = accuracy_score(y_test, y_pred)
                                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                                
                                metrics = {
                                    "Accuracy": accuracy,
                                    "Precision": precision,
                                    "Recall": recall,
                                    "F1 Score": f1
                                }
                                
                                model_info = {
                                    "name": f"{model_name} (NLP)",
                                    "model": model,
                                    "metrics": metrics,
                                    "type": "nlp"
                                }
                                st.session_state.trained_models.append(model_info)
                        
                        st.success(f"✅ Trained {len(models)} NLP models!")
                        
                    elif nlp_task == "Topic Modeling":
                        st.markdown("### 🔍 Topic Modeling with LDA")
                        
                        n_topics = st.slider("Number of Topics", 2, 10, 5)
                        
                        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
                        lda.fit(X)
                        
                        st.success("✅ Topic modeling completed!")
                        
                        # Display topics
                        st.markdown("### 📋 Discovered Topics")
                        
                        feature_names = vectorizer.get_feature_names_out()
                        for topic_idx, topic in enumerate(lda.components_):
                            top_words_idx = topic.argsort()[-10:][::-1]
                            top_words = [feature_names[i] for i in top_words_idx]
                            
                            with st.expander(f"Topic {topic_idx + 1}"):
                                st.write(", ".join(top_words))
                        
                        model_info = {
                            "name": "LDA Topic Modeling",
                            "model": lda,
                            "metrics": {"Number of Topics": n_topics},
                            "type": "nlp"
                        }
                        st.session_state.trained_models.append(model_info)
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Model Comparison Page
elif page == "📈 Model Comparison":
    st.markdown("<h2>📈 Compare Model Performance</h2>", unsafe_allow_html=True)
    
    if not st.session_state.trained_models:
        st.warning("⚠️ No trained models to compare. Please train some models first!")
    else:
        st.success(f"🎯 Comparing {len(st.session_state.trained_models)} trained models")
        
        # Filter models by type
        model_types = list(set([m['type'] for m in st.session_state.trained_models]))
        selected_type = st.selectbox("Filter by Model Type", ["All"] + model_types)
        
        if selected_type == "All":
            models_to_compare = st.session_state.trained_models
        else:
            models_to_compare = [m for m in st.session_state.trained_models if m['type'] == selected_type]
        
        # Create comparison dataframe
        comparison_data = []
        for model_info in models_to_compare:
            row = {"Model": model_info['name'], "Type": model_info['type']}
            row.update(model_info['metrics'])
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        st.markdown("### 📊 Performance Comparison Table")
        st.dataframe(comparison_df.set_index("Model"), use_container_width=True)
        
        # Visualizations
        st.markdown("### 📈 Performance Visualizations")
        
        metric_cols = [col for col in comparison_df.columns if col not in ["Model", "Type"]]
        
        if metric_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_metric = st.selectbox("Select Metric", metric_cols)
            
            with col2:
                chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Radar Chart"])
            
            if chart_type == "Bar Chart":
                fig = px.bar(
                    comparison_df,
                    x="Model",
                    y=selected_metric,
                    title=f"{selected_metric} Comparison",
                    color=selected_metric,
                    color_continuous_scale="Viridis",
                    text=selected_metric
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Line Chart":
                fig = px.line(
                    comparison_df,
                    x="Model",
                    y=selected_metric,
                    title=f"{selected_metric} Comparison",
                    markers=True
                )
                fig.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Radar Chart" and len(metric_cols) > 2:
                fig = go.Figure()
                
                for _, row in comparison_df.iterrows():
                    values = [row[col] for col in metric_cols if pd.notna(row[col])]
                    valid_metrics = [col for col in metric_cols if pd.notna(row[col])]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=valid_metrics,
                        fill='toself',
                        name=row['Model']
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Multi-Metric Radar Comparison",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Best model
        st.markdown("### 🏆 Best Performing Models")
        
        for metric in metric_cols:
            if comparison_df[metric].dtype in ['float64', 'int64']:
                # For metrics where higher is better (most cases)
                if metric not in ["MSE", "RMSE", "MAE", "Davies-Bouldin Index"]:
                    best_idx = comparison_df[metric].idxmax()
                else:
                    best_idx = comparison_df[metric].idxmin()
                
                best_model = comparison_df.loc[best_idx]
                
                st.markdown(f"""
                <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 10px 0;'>
                    <h4 style='color: white; margin: 0;'>Best {metric}: {best_model['Model']}</h4>
                    <p style='color: white; margin: 5px 0;'>Score: {best_model[metric]:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Download comparison
        st.markdown("### 💾 Export Results")
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison as CSV",
            data=csv,
            file_name="model_comparison.csv",
            mime="text/csv"
        )

# Test & Deploy Page
elif page == "🧪 Test & Deploy":
    st.markdown("<h2>🧪 Test Your Models</h2>", unsafe_allow_html=True)
    
    if not st.session_state.trained_models:
        st.warning("⚠️ No trained models available. Please train some models first!")
    else:
        model_names = [m['name'] for m in st.session_state.trained_models]
        selected_model_name = st.selectbox("🤖 Select Model for Testing", model_names)
        
        selected_model_info = next(m for m in st.session_state.trained_models if m['name'] == selected_model_name)
        model = selected_model_info['model']
        model_type = selected_model_info['type']
        
        st.markdown(f"**Model Type:** {model_type.upper()}")
        st.markdown("---")
        
        if model_type == "supervised":
            st.markdown("### 📝 Enter Feature Values")
            
            input_data = {}
            cols = st.columns(3)
            
            for i, feature in enumerate(st.session_state.feature_names):
                with cols[i % 3]:
                    input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f", key=f"input_{i}")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                predict_btn = st.button("🎯 Make Prediction", use_container_width=True)
            
            if predict_btn:
                try:
                    input_df = pd.DataFrame([input_data])
                    input_scaled = st.session_state.scaler.transform(input_df)
                    
                    prediction = model.predict(input_scaled)[0]
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; text-align: center;'>
                        <h2 style='color: white; margin: 0;'>🎯 Prediction Result</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    if st.session_state.problem_type == "Classification":
                        st.markdown(f"""
                        <div style='background-color: rgba(255, 255, 255, 0.95); padding: 40px; border-radius: 15px; text-align: center;'>
                            <h1 style='color: #667eea; font-size: 3em; margin: 0;'>{int(prediction)}</h1>
                            <p style='color: #666; font-size: 1.2em;'>Predicted Class</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background-color: rgba(255, 255, 255, 0.95); padding: 40px; border-radius: 15px; text-align: center;'>
                            <h1 style='color: #667eea; font-size: 3em; margin: 0;'>{prediction:.4f}</h1>
                            <p style='color: #666; font-size: 1.2em;'>Predicted Value</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with st.expander("📊 View Input Values"):
                        st.json(input_data)
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
        
        elif model_type == "nlp":
            st.markdown("### 📝 Enter Text for Prediction")
            
            text_input = st.text_area("Input Text", height=150, placeholder="Enter your text here...")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                predict_btn = st.button("🎯 Analyze Text", use_container_width=True)
            
            if predict_btn and text_input:
                try:
                    if st.session_state.vectorizer is not None:
                        text_vectorized = st.session_state.vectorizer.transform([text_input])
                        
                        if hasattr(model, 'predict'):
                            prediction = model.predict(text_vectorized)[0]
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; text-align: center;'>
                                <h2 style='color: white; margin: 0;'>📊 Analysis Result</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div style='background-color: rgba(255, 255, 255, 0.95); padding: 40px; border-radius: 15px; text-align: center;'>
                                <h1 style='color: #667eea; font-size: 3em; margin: 0;'>{prediction}</h1>
                                <p style='color: #666; font-size: 1.2em;'>Predicted Category</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(text_vectorized)[0]
                                
                                st.markdown("### 📊 Confidence Scores")
                                prob_df = pd.DataFrame({
                                    'Class': range(len(proba)),
                                    'Probability': proba
                                })
                                
                                fig = px.bar(prob_df, x='Class', y='Probability', 
                                           title="Prediction Probabilities",
                                           color='Probability',
                                           color_continuous_scale='Viridis')
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("This model doesn't support predictions. It's a topic modeling algorithm.")
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing text: {str(e)}")
        
        elif model_type == "clustering":
            st.info("ℹ️ Clustering models don't make individual predictions. Use the clustering page to see cluster assignments for your entire dataset.")
        
        # Model Info
        st.markdown("---")
        st.markdown("### 📊 Model Performance Metrics")
        
        metrics = selected_model_info['metrics']
        cols = st.columns(len(metrics))
        
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            with cols[i]:
                st.markdown(f"""
                <div style='background-color: rgba(255, 255, 255, 0.15); padding: 20px; border-radius: 10px; text-align: center;'>
                    <h3 style='color: white; font-size: 2em; margin: 0;'>{metric_value:.4f if isinstance(metric_value, float) else metric_value}</h3>
                    <p style='color: white; margin: 5px 0;'>{metric_name}</p>
                </div>
                """, unsafe_allow_html=True)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Quick Guide")
st.sidebar.markdown("""
1. **Import Data** - Upload CSV or import from Kaggle
2. **Choose Task** - Supervised, Clustering, or NLP
3. **Train Models** - Select and train algorithms
4. **Compare** - Analyze performance metrics
5. **Test** - Make predictions on new data
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Settings")

if st.sidebar.button("🗑️ Clear All Models"):
    st.session_state.trained_models = []
    st.success("✅ All models cleared!")
    st.rerun()

if st.sidebar.button("🔄 Reset Application"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("✅ Application reset!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;'>
    <p style='color: white; margin: 0; text-align: center;'>
        Made with ❤️ using Streamlit
    </p>
</div>
""", unsafe_allow_html=True)