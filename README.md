# Smart City Traffic Incident Prediction

## 📌 Project Overview
This project focuses on predicting and managing traffic incidents in a smart city environment using **Amazon SageMaker, AWS S3, and XGBoost**. The goal was to develop a machine learning model that can accurately classify traffic incidents based on historical and real-time data.

### 🔍 Problem Statement
Traffic congestion and accidents are major concerns in urban areas. The challenge was to build a **predictive model** that could classify traffic conditions as either "Incident" or "No Incident." However, the dataset was **highly imbalanced**, making it difficult for traditional models to detect real incidents accurately.

## 🛠️ Technologies Used
- **Amazon SageMaker**: For training, deploying, and hosting the machine learning model.
- **AWS S3**: Used as the data storage for training and validation datasets.
- **XGBoost**: A powerful gradient boosting algorithm used to build the classifier.
- **Python & Pandas**: For data processing and analysis.
- **Scikit-Learn**: Used for model evaluation and performance metrics.
- **Plotly**: For visualization of results.

## 🚧 Challenges Faced
1. **Class Imbalance**
   - The dataset had significantly more "No Incident" cases than "Incident" cases.
   - Traditional models struggled to detect real incidents due to the imbalance.
   - **Solution:** Used **cost-sensitive learning** (scale_pos_weight) and resampling strategies.

2. **Model Deployment Issues**
   - Encountered an issue where the deployed model endpoint was not accessible.
   - The error message indicated **missing model artifacts in the S3 bucket**.
   - **Solution:** Verified the correct model path in S3 (`model.tar.gz`), updated the deployment script, and redeployed the endpoint.

3. **Inference Errors**
   - Encountered issues with data serialization and incorrect content type.
   - Initial errors in the JSON response format led to failed predictions.
   - **Solution:** Corrected request format, ensured CSV serialization, and handled response parsing properly.

## 📊 Results
| Metric      | Score  |
|------------|--------|
| Accuracy   | **97.5%** |
| Precision  | **95.1%** |
| Recall     | **100%** |
| F1-Score   | **97.49%** |

### 🔹 Confusion Matrix:
|               | No Incident | Incident |
|--------------|------------|----------|
| **No Incident** | 98         | 5        |
| **Incident**    | 0          | 97       |

```

## 🚀 How to Run the Project
1. **Clone the Repository:**
   ```sh
   git clone https://github.com/your-username/smart-city-traffic-ml.git
   cd smart-city-traffic-ml
   ```
2. **Set Up AWS Credentials:**
   - Configure AWS CLI with the correct permissions for S3 and SageMaker.
3. **Run the Jupyter Notebook:**
   - Open the notebook
   - Follow the steps for data loading, preprocessing, training, and evaluation.
4. **Deploy the Model:**
   ```python
   predictor = xgboost.deploy(
       initial_instance_count=1,
       instance_type="ml.m5.large",
       endpoint_name="xgboost-traffic-predictor"
   )
   ```

## 📌 Future Improvements
- **Enhance Model Robustness** by experimenting with different resampling techniques.
- **Improve Real-Time Predictions** by integrating with live traffic data APIs.
- **Deploy a Scalable API** using AWS Lambda & API Gateway for real-time inference.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

