# 🩺 Diabetes Prediction ML Model

An interactive machine learning web application for diabetes prediction built with Streamlit. This application provides comprehensive data analysis, multiple ML algorithms, and an intuitive prediction interface for healthcare professionals and researchers.

## 🌟 Features

### 📊 Data Analysis & Visualization
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
   cd diabetes-prediction-ml
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
- Review the Streamlit documentation for UI-related questions

## 🙏 Acknowledgments

- **Scikit-learn** for the diabetes dataset and ML algorithms
- **Streamlit** for the excellent web framework
- **Plotly** for interactive visualizations
- **Healthcare community** for inspiring this predictive modeling approach

---

**Made with ❤️ for healthcare innovation and machine learning education**

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










Sample Output 📊

You can view the sample output here: https://docs.google.com/document/d/1uSghDANdLfcDh4JzSpq9kivx0DrOlrd-sxvTS8dhgsk/edit?usp=drivesdk 📄

Author 👩‍💻
Reaishma N 

