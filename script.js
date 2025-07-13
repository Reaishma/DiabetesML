// Diabetes Prediction ML Model - JavaScript Implementation
// Mobile-responsive Streamlit 

class DiabetesPredictionApp {
    constructor() {
        this.data = null;
        this.models = null;
        this.scaler = null;
        this.modelsTrained = false;
        this.currentPage = 'overview';
        
        this.init();
    }
    
    init() {
        this.loadSampleData();
        this.setupEventListeners();
        this.setupSliders();
        this.showPage('overview');
        this.populateDataOverview();
    }
    
    // Generate sample diabetes dataset
    loadSampleData() {
        const n_samples = 442;
        const features = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'];
        
        this.data = {
            features: features,
            samples: [],
            target: []
        };
        
        // Generate random standardized data
        for (let i = 0; i < n_samples; i++) {
            const sample = {};
            features.forEach(feature => {
                if (feature === 'sex') {
                    sample[feature] = Math.random() > 0.5 ? 1 : -1;
                } else {
                    sample[feature] = (Math.random() - 0.5) * 4;
                }
            });
            
            // Generate target based on feature combination
            const target_score = sample.bmi * 0.3 + sample.bp * 0.2 + sample.s6 * 0.4 + 
                               sample.age * 0.1 + Math.random() * 0.5;
            sample.diabetes = target_score > 0 ? 1 : 0;
            
            this.data.samples.push(sample);
            this.data.target.push(sample.diabetes);
        }
        
        console.log('Sample data loaded:', this.data.samples.length, 'samples');
    }
    
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = e.target.getAttribute('data-page');
                this.showPage(page);
                this.closeMobileMenu();
            });
        });
        
        // Mobile menu functionality
        const mobileMenuBtn = document.getElementById('mobile-menu-btn');
        const mobileOverlay = document.getElementById('mobile-overlay');
        
        mobileMenuBtn.addEventListener('click', () => {
            this.toggleMobileMenu();
        });
        
        mobileOverlay.addEventListener('click', () => {
            this.closeMobileMenu();
        });
        
        // Close mobile menu on window resize
        window.addEventListener('resize', () => {
            if (window.innerWidth > 768) {
                this.closeMobileMenu();
            }
            // Update charts on resize for responsiveness
            this.updateChartsOnResize();
        });
        
        // Train models button
        document.getElementById('train-models-btn').addEventListener('click', () => {
            this.trainModels();
        });
        
        // Prediction button
        document.getElementById('predict-btn').addEventListener('click', () => {
            this.makePrediction();
        });
        
        // Feature selector for EDA
        document.getElementById('feature-select').addEventListener('change', (e) => {
            this.updateFeatureBoxplot(e.target.value);
        });
    }
    
    setupSliders() {
        const sliders = ['age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'];
        
        sliders.forEach(slider => {
            const sliderElement = document.getElementById(`${slider}-slider`);
            const valueElement = document.getElementById(`${slider}-value`);
            
            if (sliderElement && valueElement) {
                sliderElement.addEventListener('input', (e) => {
                    valueElement.textContent = parseFloat(e.target.value).toFixed(1);
                });
            }
        });
    }
    
    toggleMobileMenu() {
        const sidebar = document.querySelector('.sidebar');
        const overlay = document.getElementById('mobile-overlay');
        const isOpen = sidebar.classList.contains('open');
        
        if (isOpen) {
            this.closeMobileMenu();
        } else {
            this.openMobileMenu();
        }
    }
    
    openMobileMenu() {
        const sidebar = document.querySelector('.sidebar');
        const overlay = document.getElementById('mobile-overlay');
        
        sidebar.classList.add('open');
        overlay.style.display = 'block';
        document.body.style.overflow = 'hidden';
    }
    
    closeMobileMenu() {
        const sidebar = document.querySelector('.sidebar');
        const overlay = document.getElementById('mobile-overlay');
        
        sidebar.classList.remove('open');
        overlay.style.display = 'none';
        document.body.style.overflow = '';
    }
    
    showPage(pageName) {
        // Update navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[data-page="${pageName}"]`).classList.add('active');
        
        // Update page content
        document.querySelectorAll('.page').forEach(page => {
            page.classList.remove('active');
        });
        document.getElementById(`${pageName}-page`).classList.add('active');
        
        this.currentPage = pageName;
        
        // Load page-specific content
        switch(pageName) {
            case 'overview':
                this.populateDataOverview();
                break;
            case 'eda':
                this.createEDACharts();
                break;
            case 'training':
                this.updateTrainingPage();
                break;
            case 'prediction':
                this.updatePredictionPage();
                break;
        }
    }
    
    populateDataOverview() {
        // Update metrics
        const diabetesCases = this.data.target.filter(x => x === 1).length;
        const noDiabetesCases = this.data.target.filter(x => x === 0).length;
        
        document.getElementById('total-samples').textContent = this.data.samples.length;
        document.getElementById('features-count').textContent = this.data.features.length;
        document.getElementById('diabetes-cases').textContent = diabetesCases;
        document.getElementById('no-diabetes-cases').textContent = noDiabetesCases;
        
        // Populate sample data table
        const tbody = document.getElementById('dataset-tbody');
        tbody.innerHTML = '';
        
        // Show first 10 samples
        for (let i = 0; i < Math.min(10, this.data.samples.length); i++) {
            const sample = this.data.samples[i];
            const row = document.createElement('tr');
            
            this.data.features.forEach(feature => {
                const cell = document.createElement('td');
                cell.textContent = sample[feature].toFixed(3);
                row.appendChild(cell);
            });
            
            // Add diabetes column
            const diabetesCell = document.createElement('td');
            diabetesCell.textContent = sample.diabetes;
            row.appendChild(diabetesCell);
            
            tbody.appendChild(row);
        }
    }
    
    createEDACharts() {
        this.createTargetDistributionChart();
        this.createFeatureDistributionsChart();
        this.createCorrelationMatrix();
        this.updateFeatureBoxplot('age');
    }
    
    createTargetDistributionChart() {
        const diabetesCases = this.data.target.filter(x => x === 1).length;
        const noDiabetesCases = this.data.target.filter(x => x === 0).length;
        
        const data = [{
            labels: ['No Diabetes', 'Diabetes'],
            values: [noDiabetesCases, diabetesCases],
            type: 'pie',
            marker: {
                colors: ['#28A745', '#DC3545']
            }
        }];
        
        const isMobile = window.innerWidth < 768;
        const layout = {
            title: 'Distribution of Diabetes Cases',
            height: isMobile ? 300 : 400,
            responsive: true,
            font: { size: isMobile ? 10 : 12 }
        };
        
        const config = {
            responsive: true,
            displayModeBar: !isMobile
        };
        
        Plotly.newPlot('target-distribution-chart', data, layout, config);
    }
    
    createFeatureDistributionsChart() {
        const features = this.data.features;
        const traces = [];
        
        features.forEach((feature, index) => {
            const values = this.data.samples.map(sample => sample[feature]);
            
            traces.push({
                x: values,
                type: 'histogram',
                name: feature,
                xaxis: `x${index + 1}`,
                yaxis: `y${index + 1}`,
                showlegend: false
            });
        });
        
        const isMobile = window.innerWidth < 768;
        const layout = {
            title: 'Feature Distributions',
            height: isMobile ? 1200 : 800,
            responsive: true,
            font: { size: isMobile ? 8 : 10 },
            grid: {
                rows: isMobile ? features.length : Math.ceil(features.length / 2),
                columns: isMobile ? 1 : 2,
                subplots: features.map((_, i) => [`x${i + 1}`, `y${i + 1}`])
            }
        };
        
        // Add subplot annotations
        features.forEach((feature, index) => {
            layout[`xaxis${index + 1}`] = { title: feature };
            layout[`yaxis${index + 1}`] = { title: 'Count' };
        });
        
        const config = {
            responsive: true,
            displayModeBar: !isMobile
        };
        
        Plotly.newPlot('feature-distributions-chart', traces, layout, config);
    }
    
    createCorrelationMatrix() {
        const features = this.data.features;
        const correlationMatrix = this.calculateCorrelationMatrix();
        
        const data = [{
            z: correlationMatrix,
            x: features,
            y: features,
            type: 'heatmap',
            colorscale: 'RdBu',
            reversescale: true,
            showscale: true
        }];
        
        const isMobile = window.innerWidth < 768;
        const layout = {
            title: 'Feature Correlation Matrix',
            height: isMobile ? 350 : 500,
            responsive: true,
            font: { size: isMobile ? 8 : 10 },
            xaxis: { 
                title: 'Features',
                tickangle: isMobile ? -45 : 0,
                tickfont: { size: isMobile ? 6 : 8 }
            },
            yaxis: { 
                title: 'Features',
                tickfont: { size: isMobile ? 6 : 8 }
            }
        };
        
        const config = {
            responsive: true,
            displayModeBar: !isMobile
        };
        
        Plotly.newPlot('correlation-matrix-chart', data, layout, config);
    }
    
    calculateCorrelationMatrix() {
        const features = this.data.features;
        const matrix = [];
        
        features.forEach((feature1, i) => {
            const row = [];
            features.forEach((feature2, j) => {
                const values1 = this.data.samples.map(sample => sample[feature1]);
                const values2 = this.data.samples.map(sample => sample[feature2]);
                const correlation = this.pearsonCorrelation(values1, values2);
                row.push(correlation);
            });
            matrix.push(row);
        });
        
        return matrix;
    }
    
    pearsonCorrelation(x, y) {
        const n = x.length;
        const sum_x = x.reduce((a, b) => a + b, 0);
        const sum_y = y.reduce((a, b) => a + b, 0);
        const sum_xy = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
        const sum_x_sq = x.reduce((acc, xi) => acc + xi * xi, 0);
        const sum_y_sq = y.reduce((acc, yi) => acc + yi * yi, 0);
        
        const numerator = n * sum_xy - sum_x * sum_y;
        const denominator = Math.sqrt((n * sum_x_sq - sum_x * sum_x) * (n * sum_y_sq - sum_y * sum_y));
        
        return denominator === 0 ? 0 : numerator / denominator;
    }
    
    updateFeatureBoxplot(selectedFeature) {
        const diabetesValues = [];
        const noDiabetesValues = [];
        
        this.data.samples.forEach(sample => {
            if (sample.diabetes === 1) {
                diabetesValues.push(sample[selectedFeature]);
            } else {
                noDiabetesValues.push(sample[selectedFeature]);
            }
        });
        
        const data = [
            {
                y: noDiabetesValues,
                type: 'box',
                name: 'No Diabetes',
                marker: { color: '#28A745' }
            },
            {
                y: diabetesValues,
                type: 'box',
                name: 'Diabetes',
                marker: { color: '#DC3545' }
            }
        ];
        
        const isMobile = window.innerWidth < 768;
        const layout = {
            title: `${selectedFeature} Distribution by Diabetes Status`,
            yaxis: { title: selectedFeature },
            height: isMobile ? 300 : 400,
            responsive: true,
            font: { size: isMobile ? 10 : 12 }
        };
        
        const config = {
            responsive: true,
            displayModeBar: !isMobile
        };
        
        Plotly.newPlot('feature-boxplot-chart', data, layout, config);
    }
    
    updateChartsOnResize() {
        // Redraw charts when window is resized to maintain responsiveness
        if (this.currentPage === 'eda') {
            setTimeout(() => {
                this.createEDACharts();
            }, 300);
        }
    }
    
    async trainModels() {
        const trainBtn = document.getElementById('train-models-btn');
        const status = document.getElementById('training-status');
        
        trainBtn.disabled = true;
        status.innerHTML = '<div class="spinner"></div>Training models...';
        status.className = 'training-status loading';
        
        await this.delay(1000);
        
        // Initialize models
        this.models = {
            logisticRegression: new LogisticRegressionModel(),
            randomForest: new RandomForestModel(),
            svm: new SVMModel()
        };
        
        // Prepare training data
        const X = this.data.samples.map(sample => 
            this.data.features.map(feature => sample[feature])
        );
        const y = this.data.target;
        
        // Train-test split
        const splitIndex = Math.floor(X.length * 0.8);
        const X_train = X.slice(0, splitIndex);
        const y_train = y.slice(0, splitIndex);
        const X_test = X.slice(splitIndex);
        const y_test = y.slice(splitIndex);
        
        // Train models
        await this.models.logisticRegression.fit(X_train, y_train);
        await this.models.randomForest.fit(X_train, y_train);
        await this.models.svm.fit(X_train, y_train);
        
        // Store test data for evaluation
        this.testData = { X_test, y_test };
        
        this.modelsTrained = true;
        
        status.innerHTML = '✅ Models trained successfully!';
        status.className = 'training-status success';
        trainBtn.disabled = false;
        
        // Show results
        this.displayTrainingResults();
    }
    
    displayTrainingResults() {
        document.getElementById('training-results').style.display = 'block';
        this.updatePerformanceTable();
        this.createPerformanceChart();
        this.createConfusionMatrices();
        this.createROCCurves();
        this.createFeatureImportanceChart();
    }
    
    updatePerformanceTable() {
        const tbody = document.getElementById('performance-tbody');
        tbody.innerHTML = '';
        
        const modelNames = ['Logistic Regression', 'Random Forest', 'SVM'];
        const modelKeys = ['logisticRegression', 'randomForest', 'svm'];
        
        modelKeys.forEach((key, index) => {
            const model = this.models[key];
            const predictions = model.predict(this.testData.X_test);
            
            const accuracy = this.calculateAccuracy(this.testData.y_test, predictions);
            const precision = this.calculatePrecision(this.testData.y_test, predictions);
            const recall = this.calculateRecall(this.testData.y_test, predictions);
            const f1Score = this.calculateF1Score(this.testData.y_test, predictions);
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${modelNames[index]}</td>
                <td>${accuracy.toFixed(3)}</td>
                <td>${precision.toFixed(3)}</td>
                <td>${recall.toFixed(3)}</td>
                <td>${f1Score.toFixed(3)}</td>
            `;
            tbody.appendChild(row);
        });
    }
    
    createPerformanceChart() {
        const modelNames = ['Logistic Regression', 'Random Forest', 'SVM'];
        const modelKeys = ['logisticRegression', 'randomForest', 'svm'];
        const metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score'];
        
        const traces = metrics.map(metric => ({
            x: modelNames,
            y: modelKeys.map(key => {
                const model = this.models[key];
                const predictions = model.predict(this.testData.X_test);
                
                switch(metric) {
                    case 'Accuracy': return this.calculateAccuracy(this.testData.y_test, predictions);
                    case 'Precision': return this.calculatePrecision(this.testData.y_test, predictions);
                    case 'Recall': return this.calculateRecall(this.testData.y_test, predictions);
                    case 'F1-Score': return this.calculateF1Score(this.testData.y_test, predictions);
                }
            }),
            type: 'bar',
            name: metric
        }));
        
        const isMobile = window.innerWidth < 768;
        const layout = {
            title: 'Model Performance Comparison',
            xaxis: { title: 'Models' },
            yaxis: { title: 'Score' },
            height: isMobile ? 300 : 400,
            responsive: true,
            font: { size: isMobile ? 10 : 12 }
        };
        
        const config = {
            responsive: true,
            displayModeBar: !isMobile
        };
        
        Plotly.newPlot('performance-chart', traces, layout, config);
    }
    
    createConfusionMatrices() {
        const modelNames = ['Logistic Regression', 'Random Forest', 'SVM'];
        const modelKeys = ['logisticRegression', 'randomForest', 'svm'];
        const containers = ['confusion-lr', 'confusion-rf', 'confusion-svm'];
        
        modelKeys.forEach((key, index) => {
            const model = this.models[key];
            const predictions = model.predict(this.testData.X_test);
            const cm = this.calculateConfusionMatrix(this.testData.y_test, predictions);
            
            const data = [{
                z: cm,
                x: ['No Diabetes', 'Diabetes'],
                y: ['No Diabetes', 'Diabetes'],
                type: 'heatmap',
                colorscale: 'Blues',
                showscale: false
            }];
            
            const isMobile = window.innerWidth < 768;
            const layout = {
                title: `${modelNames[index]} Confusion Matrix`,
                height: isMobile ? 250 : 300,
                responsive: true,
                font: { size: isMobile ? 8 : 10 }
            };
            
            const config = {
                responsive: true,
                displayModeBar: !isMobile
            };
            
 