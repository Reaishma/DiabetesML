import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction ML Model",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🩺 Diabetes Prediction ML Model")
st.markdown("**An interactive machine learning application for diabetes prediction with comprehensive data analysis**")

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the diabetes dataset for classification"""
    # Load the regression diabetes dataset
    diabetes = load_diabetes()
    
    # Convert to DataFrame
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    
    # Convert target to binary classification (diabetes/no diabetes)
    # Using median as threshold for binary classification
    target_median = np.median(diabetes.target)
    df['diabetes'] = (diabetes.target > target_median).astype(int)
    
    # Add feature descriptions
    feature_descriptions = {
        'age': 'Age of the patient',
        'sex': 'Sex of the patient (1 = male, -1 = female)',
        'bmi': 'Body Mass Index',
        'bp': 'Average Blood Pressure',
        's1': 'Total Serum Cholesterol',
        's2': 'Low-Density Lipoproteins',
        's3': 'High-Density Lipoproteins',
        's4': 'Total Cholesterol / HDL',
        's5': 'Log of Serum Triglycerides',
        's6': 'Blood Sugar Level'
    }
    
    return df, feature_descriptions

@st.cache_data
def preprocess_data(df):
    """Preprocess the data for machine learning"""
    # Separate features and target
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

@st.cache_data
def train_models(X_train_scaled, y_train):
    """Train multiple ML models"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test_scaled, y_test):
    """Evaluate model performance"""
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    return results

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix using Plotly"""
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title=f'Confusion Matrix - {model_name}',
        labels=dict(x="Predicted", y="Actual"),
        x=['No Diabetes', 'Diabetes'],
        y=['No Diabetes', 'Diabetes']
    )
    return fig

def plot_roc_curve(models_results, y_test):
    """Plot ROC curves for all models"""
    fig = go.Figure()
    
    for name, results in models_results.items():
        fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
        auc_score = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{name} (AUC = {auc_score:.3f})',
            line=dict(width=2)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Feature Importance - {model_name}',
        labels={'importance': 'Importance Score', 'feature': 'Features'}
    )
    
    return fig

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Data Overview", "Exploratory Data Analysis", "Model Training & Evaluation", "Prediction Interface"]
)

# Load data
try:
    df, feature_descriptions = load_and_prepare_data()
    st.session_state.data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Data Overview Page
if page == "Data Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Diabetes Cases", df['diabetes'].sum())
    with col4:
        st.metric("No Diabetes Cases", len(df) - df['diabetes'].sum())
    
    st.subheader("Dataset Sample")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Feature Descriptions")
    desc_df = pd.DataFrame(list(feature_descriptions.items()), columns=['Feature', 'Description'])
    st.dataframe(desc_df, use_container_width=True)
    
    st.subheader("Missing Values Check")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        st.success("✅ No missing values found in the dataset!")
    else:
        st.dataframe(missing_values[missing_values > 0])

# Exploratory Data Analysis Page
elif page == "Exploratory Data Analysis":
    st.header("🔍 Exploratory Data Analysis")
    
    # Target distribution
    st.subheader("Target Variable Distribution")
    fig_target = px.pie(
        values=[len(df) - df['diabetes'].sum(), df['diabetes'].sum()],
        names=['No Diabetes', 'Diabetes'],
        title='Distribution of Diabetes Cases'
    )
    st.plotly_chart(fig_target, use_container_width=True)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    feature_cols = [col for col in df.columns if col != 'diabetes']
    
    # Create subplot for feature distributions
    rows = (len(feature_cols) + 2) // 3
    fig_dist = make_subplots(
        rows=rows, cols=3,
        subplot_titles=feature_cols,
        vertical_spacing=0.08
    )
    
    for i, feature in enumerate(feature_cols):
        row = i // 3 + 1
        col = i % 3 + 1
        
        fig_dist.add_trace(
            go.Histogram(x=df[feature], name=feature, showlegend=False),
            row=row, col=col
        )
    
    fig_dist.update_layout(height=300*rows, title_text="Feature Distributions")
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Correlation matrix
    st.subheader("Feature Correlation Matrix")
    corr_matrix = df.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Box plots by diabetes status
    st.subheader("Feature Distributions by Diabetes Status")
    selected_feature = st.selectbox("Select feature to analyze:", feature_cols)
    
    fig_box = px.box(
        df, 
        x='diabetes', 
        y=selected_feature,
        title=f'{selected_feature} Distribution by Diabetes Status',
        labels={'diabetes': 'Diabetes Status (0=No, 1=Yes)'}
    )
    st.plotly_chart(fig_box, use_container_width=True)

# Model Training & Evaluation Page
elif page == "Model Training & Evaluation":
    st.header("🤖 Model Training & Evaluation")
    
    if st.button("Train Models", type="primary"):
        with st.spinner("Training models..."):
            # Preprocess data
            X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = preprocess_data(df)
            
            # Train models
            trained_models = train_models(X_train_scaled, y_train)
            
            # Evaluate models
            results = evaluate_models(trained_models, X_test_scaled, y_test)
            
            # Store in session state
            st.session_state.trained_models = trained_models
            st.session_state.scaler = scaler
            st.session_state.results = results
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.models_trained = True
            
        st.success("✅ Models trained successfully!")
        st.rerun()
    
    if st.session_state.models_trained:
        results = st.session_state.results
        
        # Model performance comparison
        st.subheader("Model Performance Comparison")
        
        performance_data = []
        for model_name, result in results.items():
            performance_data.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score']
            })
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
        
        # Metrics visualization
        fig_metrics = px.bar(
            performance_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
            x='Model',
            y='Score',
            color='Metric',
            title='Model Performance Metrics Comparison',
            barmode='group'
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        
        cols = st.columns(3)
        for i, (model_name, result) in enumerate(results.items()):
            with cols[i]:
                fig_cm = plot_confusion_matrix(result['confusion_matrix'], model_name)
                st.plotly_chart(fig_cm, use_container_width=True)
        
        # ROC curves
        st.subheader("ROC Curves")
        fig_roc = plot_roc_curve(results, st.session_state.y_test)
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance Analysis")
        
        feature_names = [col for col in df.columns if col != 'diabetes']
        
        for model_name, model in st.session_state.trained_models.items():
            if model_name == 'Random Forest':  # Show feature importance for Random Forest
                fig_importance = plot_feature_importance(model, feature_names, model_name)
                if fig_importance:
                    st.plotly_chart(fig_importance, use_container_width=True)
                break

# Prediction Interface Page
elif page == "Prediction Interface":
    st.header("🔮 Diabetes Prediction Interface")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first in the 'Model Training & Evaluation' page.")
    else:
        st.subheader("Enter Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                          help="Standardized age value")
            sex = st.selectbox("Sex", options=[-1, 1], format_func=lambda x: "Female" if x == -1 else "Male")
            bmi = st.slider("BMI", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                          help="Standardized BMI value")
            bp = st.slider("Blood Pressure", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                         help="Standardized blood pressure value")
            s1 = st.slider("Total Serum Cholesterol", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                         help="Standardized cholesterol value")
        
        with col2:
            s2 = st.slider("Low-Density Lipoproteins", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                         help="Standardized LDL value")
            s3 = st.slider("High-Density Lipoproteins", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                         help="Standardized HDL value")
            s4 = st.slider("Total Cholesterol / HDL", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                         help="Standardized cholesterol ratio")
            s5 = st.slider("Log of Serum Triglycerides", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                         help="Standardized triglycerides value")
            s6 = st.slider("Blood Sugar Level", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                         help="Standardized blood sugar value")
        
        if st.button("Predict Diabetes Risk", type="primary"):
            # Prepare input data
            input_data = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])
            
            st.subheader("Prediction Results")
            
            # Make predictions with all models
            cols = st.columns(3)
            
            for i, (model_name, model) in enumerate(st.session_state.trained_models.items()):
                with cols[i]:
                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0]
                    
                    # Display prediction
                    if prediction == 1:
                        st.error(f"**{model_name}**: High Diabetes Risk")
                        risk_level = "High Risk"
                        color = "red"
                    else:
                        st.success(f"**{model_name}**: Low Diabetes Risk")
                        risk_level = "Low Risk"
                        color = "green"
                    
                    # Display confidence
                    confidence = max(probability) * 100
                    st.metric(
                        label="Confidence",
                        value=f"{confidence:.1f}%"
                    )
                    
                    # Probability breakdown
                    st.write("**Probability Breakdown:**")
                    st.write(f"No Diabetes: {probability[0]:.3f}")
                    st.write(f"Diabetes: {probability[1]:.3f}")
            
            # Summary
            st.subheader("Summary")
            predictions = []
            for model_name, model in st.session_state.trained_models.items():
                pred = model.predict(input_data)[0]
                predictions.append(pred)
            
            majority_vote = 1 if sum(predictions) >= 2 else 0
            
            if majority_vote == 1:
                st.error("🚨 **Majority Vote: High Diabetes Risk** - Please consult with a healthcare professional.")
            else:
                st.success("✅ **Majority Vote: Low Diabetes Risk** - Continue maintaining a healthy lifestyle.")
            
            # Feature contribution (for Random Forest)
            if 'Random Forest' in st.session_state.trained_models:
                st.subheader("Feature Contribution Analysis")
                rf_model = st.session_state.trained_models['Random Forest']
                feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
                
                # Get feature importances
                importances = rf_model.feature_importances_
                input_values = input_data[0]
                
                # Calculate contribution scores
                contributions = importances * np.abs(input_values)
                contribution_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': input_values,
                    'Importance': importances,
                    'Contribution': contributions
                }).sort_values('Contribution', ascending=False)
                
                fig_contrib = px.bar(
                    contribution_df,
                    x='Feature',
                    y='Contribution',
                    title='Feature Contribution to Prediction',
                    labels={'Contribution': 'Contribution Score'}
                )
                st.plotly_chart(fig_contrib, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🩺 Diabetes Prediction ML Model | Built with Streamlit</p>
        <p><em>Note: This tool is for educational purposes only and should not replace professional medical advice.</em></p>
    </div>
    """,
    unsafe_allow_html=True
)