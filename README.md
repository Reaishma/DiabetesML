# 🩺 Diabetes Prediction ML Model

**Diabetes Ml on streamlit - https://diabetesml-uzvcstbd8mmxtcnuwbx5xr.streamlit.app/**

This is a machine learning application for diabetes prediction. The application provides an interactive interface for data analysis, model training, and prediction using various machine learning algorithms.

![overview](https://github.com/Reaishma/DiabetesML/blob/main/Screenshot_20250904-163203_1.jpg)

# View Live Demo 

 **view demo** https://reaishma.github.io/DiabetesML/

## System Architecture

### Frontend Architecture
- **Primary Framework**: Streamlit - chosen for its simplicity in creating interactive data science applications
- **Web Version**: Complete HTML/CSS/JavaScript implementation available
- **Layout**: Wide layout with expandable sidebar for better user experience on larger screens
- **Visualization**: Dual approach using both Plotly (for interactive charts) and Matplotlib/Seaborn (for statistical plots)
- **State Management**: Streamlit session state to persist model training status and data loading state

### Backend Architecture
- **Runtime**: Python-based single-threaded application
- **Data Processing**: Pandas and NumPy for data manipulation and numerical computations
- **Machine Learning Pipeline**: Scikit-learn for model training, evaluation, and preprocessing
- **Caching Strategy**: Streamlit's `@st.cache_data` decorator for data loading optimization


## 🌟 Features

### 📊 Data Analysis & Visualization

![Data analysis](https://github.com/Reaishma/DiabetesML/blob/main/Screenshot_20250904-163220_1.jpg)

- **Interactive Dataset Overview**: Complete statistics, feature descriptions, and data quality checks
- **Exploratory Data Analysis**: Dynamic visualizations including correlation matrices, distribution plots, and feature analysis
- **Real-time Insights**: Interactive charts powered by Plotly for better data understanding

### 🤖 Machine Learning Pipeline
- **Multiple Algorithms**: 
  - Logistic Regression (linear baseline)
  - Random Forest Classifier (ensemble method)
  - Support Vector Machine (non-linear classifier)
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, ROC curves, and confusion matrices
- **Feature Importance Analysis**: Understanding which factors contribute most to diabetes prediction

### 🔮 Interactive Prediction Interface
- **User-friendly Input**: Sliders and selectors for all medical parameters
- **Multi-model Predictions**: Get predictions from all three trained models simultaneously
- **Confidence Scoring**: Probability estimates for risk assessment
- **Visual Risk Assessment**: Color-coded results with confidence indicators

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Internet connection for package installation

### Installation & Setup

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd diabetesml
   ```

2. **Install dependencies**
   ```bash
   uv add pandas numpy plotly matplotlib seaborn scikit-learn streamlit
   ```

3. **Run the application**
   ```bash
   streamlit run app.py --server.port 5000
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000` to access the application

## 📱 Application Interface

### Navigation
The application features a sidebar navigation with four main sections:

1. **Data Overview** - Dataset statistics and feature descriptions
2. **Exploratory Data Analysis** - Interactive visualizations and insights
3. **Model Training & Evaluation** - ML model comparison and performance metrics
4. **Prediction Interface** - Real-time diabetes risk prediction

**Key Elements:**
- Dataset metrics (total samples, features, diabetes cases)
- Data sample preview
- Statistical summary
- Feature descriptions table
- Missing values check

**Visualization Features:**
- Target variable distribution pie chart
- Feature distribution histograms
- Interactive correlation heatmap
- Box plots by diabetes status

**Performance Metrics:**
- Model comparison table
- Performance metrics bar chart
- Confusion matrices for all models
- ROC curves comparison
- Feature importance analysis

**Interactive Elements:**
- Patient parameter input sliders
- Real-time prediction results
- Confidence probability displays
- Multi-model comparison
- Risk level indicators

### Data Processing

![Diabetic prediction](https://github.com/Reaishma/DiabetesML/blob/main/Screenshot_20250904-163406_1.jpg)

- **Dataset**: Scikit-learn diabetes dataset (converted from regression to binary classification)
- **Features**: 10 physiological parameters (age, sex, BMI, blood pressure, cholesterol levels, etc.)
- **Preprocessing**: StandardScaler for feature normalization
- **Target**: Binary classification (diabetes risk: high/low)

### Machine Learning Pipeline
```
Data Loading → Preprocessing → Train/Test Split Datadel Training → Evaluation → Prediction
```

## 📈 Model Performance

### Evaluation Metrics
All models are evaluated using:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

### Cross-Validation
Models are validated using stratified cross-validation to ensure robust performance estimates across different data splits.

## key Elements 

### Data Management
- **Data Source**: Sklearn's built-in diabetes dataset (originally regression, converted to classification)
- **Data Transformation**: Converts continuous target variable to binary classification using median threshold
- **Feature Engineering**: Uses 10 baseline physiological features from the original dataset

### Machine Learning Pipeline

![machine learning](https://github.com/Reaishma/DiabetesML/blob/main/Screenshot_20250904-163242_1.jpg)

- **Preprocessing**: StandardScaler for feature normalization
- **Model Selection**: Three algorithms implemented:
  - Logistic Regression (linear baseline)
  - Random Forest Classifier (ensemble method)
  - Support Vector Machine (non-linear classifier)
- **Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, ROC curves, and confusion matrices
- **Validation**: Cross-validation scoring for robust model assessment

### Visualization Components
- **Interactive Charts**: Plotly Express and Graph Objects for dynamic visualizations
- **Statistical Plots**: Matplotlib and Seaborn for detailed statistical analysis
- **Performance Metrics**: Visual representation of model performance through various chart types

## Data Flow

1. **Data Loading**: Dataset loaded from sklearn and cached using Streamlit's caching mechanism
2. **Data Preprocessing**: Features standardized and target variable binarized
3. **Train-Test Split**: Data split for model training and evaluation
4. **Model Training**: Multiple algorithms trained simultaneously with cross-validation
5. **Evaluation**: Models evaluated using multiple metrics and visualizations
6. **User Interaction**: Results displayed through interactive Streamlit interface

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and utilities

### Visualization Libraries
- **plotly**: Interactive plotting (express and graph_objects modules)
- **matplotlib**: Static plotting
- **seaborn**: Statistical data visualization

## 🎯 Use Cases

### Healthcare Professionals
- **Risk Assessment**: Quick diabetes risk evaluation for patients
- **Decision Support**: Data-driven insights for clinical decisions
- **Patient Education**: Visual explanations of risk factors

### Researchers & Data Scientists
- **Model Comparison**: Benchmarking different ML algorithms
- **Feature Analysis**: Understanding important predictive factors
- **Data Exploration**: Interactive analysis of diabetes datasets

### Educational Purposes
- **ML Learning**: Hands-on experience with classification algorithms
- **Healthcare Analytics**: Understanding medical data analysis
- **Streamlit Development**: Learning interactive web app creation

## 🔧 Customization Options

### Adding New Models
To add additional ML models:
1. Import the model in the imports section
2. Add the model to the `train_models()` function
3. Ensure the model has `predict()` and `predict_proba()` methods

### Modifying Features
To use different features:
1. Update the data loading function
2. Modify feature descriptions
3. Adjust preprocessing steps if needed

### Styling & UI
Streamlit configuration can be modified in `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### Code Quality
- **Error Handling**: Comprehensive try-catch blocks for robust operation
- **Caching**: Streamlit caching for improved performance
- **Documentation**: Detailed docstrings and comments
- **Modularity**: Well-organized functions for maintainability

### Performance Optimizations
- Data caching to prevent redundant loading
- Session state management to avoid re-training models unnecessarily
- Efficient numpy/pandas operations for data processing
- Streamlit's built-in optimization for web delivery




## 🚨 Troubleshooting

### Common Issues

**Application won't start:**
- Ensure all dependencies are installed
- Check Python version (3.11+ required)
- Verify port 5000 is available

**Models not training:**
- Check data loading function
- Ensure sufficient memory available
- Verify scikit-learn installation

**Visualizations not displaying:**
- Update Plotly to latest version
- Check browser compatibility
- Clear browser cache

## Developer🧑‍💻

 **Reaishma N**

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the troubleshooting section above


## 🙏 Acknowledgments

- **Scikit-learn** for the diabetes dataset and ML algorithms
- **Streamlit** for the excellent web framework
- **Plotly** for interactive visualizations
- **Healthcare community** for inspiring this predictive modeling approaches

### Data Source
The dataset is based on the famous diabetes dataset from scikit-learn, originally collected for regression analysis and adapted here for binary classification to predict diabetes risk levels.

## 🎓 Educational Value

This project serves as an excellent learning resource for:
- **Machine Learning**: Practical implementation of classification algorithms
- **Data Science**: End-to-end ML project workflow
- **Web Development**: Creating interactive data applications
- **Healthcare Analytics**: Applying ML to medical data
- **Python Programming**: Advanced usage of data science libraries

---

**Made with ❤️ for healthcare innovation and machine learning**






