# 🔍 BHEL Vendor Delivery Prediction System

The **BHEL Vendor Delivery Prediction System** is an platform developed to enhance procurement operations by predicting vendor delivery timelines and assessing the risk of delays. Combining domain knowledge with machine learning, it provides actionable insights to optimize planning, reduce delays, and save costs.

---

## 📌 Project Overview

This system addresses two key procurement challenges:

- ✅ **On-Time Delivery Prediction** – Identify the probability of an order being delayed.
- ✅ **Accurate Delivery Date Forecasting** – Estimate the actual delivery date for any order.

It uses over 5000 historical procurement records from emulated dataset to train and validate its models, achieving:

- 📈 **85%+ accuracy** for risk classification  
- 📆 **±4.2 days mean deviation** in delivery estimates  

---

## ⚙️ System Architecture

A modular **three-tier architecture**:

### 1. 🌐 Frontend – React.js
- Intuitive form-based UI with real-time validation
- Responsive and mobile-friendly layout
- Risk level visualizations via gauges and calendar views

### 2. 🔗 Backend – FastAPI
- RESTful APIs with JWT-based user authentication
- Pydantic-based data validation
- Middleware for CORS, logging, and error handling

### 3. 🧠 ML/Database Tier
- Dual-model ML pipeline using scikit-learn
- Feature engineering, transformation, and post-processing logic
- Business rule-based prediction adjustments

---

## 🧠 Machine Learning Overview

### 🎯 On-Time Delivery Classification
- **Model**: Gradient Boosting Classifier
- **Performance**:  
  - Accuracy: 87.2%  
  - F1 Score: 84.5%  
  - ROC-AUC: 0.91  

### ⏱️ Delivery Date Regression
- **Model**: Random Forest Regressor
- **Performance**:  
  - MAE: 4.2 days  
  - RMSE: 6.8 days  
  - R² Score: 0.76  

### 🔑 Top Predictive Features
- VendorReliabilityScore
- LeadTimeDays
- MaterialComplexityScore
- Vendor Category
- IsMonsoonSeason

---

## 🖥️ User Interface Flow

1. **Vendor Selection**: Choose existing or new vendors
2. **Prediction Mode**: Select between on-time prediction or delivery estimation
3. **Data Input**: Provide vendor and order details in dynamic form tabs
4. **Results Display**: View risk level, estimated dates, and prediction confidence

---

## 🔌 API Endpoints

### `/predict/on-time`
**Method**: POST  
**Description**: Predicts on-time delivery probability.

**Response**:
```json
{
  "order_id": "ORD123",
  "on_time_probability": 87.5,
  "prediction": "On Time",
  "risk_level": "Low"
}
```

---

### `/predict/delivery-date`
**Method**: POST  
**Description**: Estimates actual delivery date based on input parameters.

**Response**:
```json
{
  "order_id": "ORD456",
  "estimated_delivery_date": "2025-07-22",
  "confidence_days": {
    "earliest": "2025-07-20",
    "latest": "2025-07-25"
  }
}
```

## 🔮 Future Enhancements

| Short-Term Goals (Q3 2025) | Long-Term Vision (2026+) |
|----------------------------|--------------------------|
| ✅ Material-specific models | 🤖 Deep learning upgrades |
| ✅ Role-based access        | 📊 Vendor recommendation engine |
| ✅ Batch prediction mode    | 💸 Price prediction engine |
| ✅ Mobile PWA UI            | 🔗 ERP/SAP Integration |

---

## 🛠️ Installation (Coming Soon)

Detailed deployment instructions will be added for:

- React frontend setup
- FastAPI backend launch
- Model serving & hosting
- Docker / Kubernetes deployment (optional)

---

## 📂 Project Structure

```
├── frontend/                # React.js frontend
├── Backend/                 # FastAPI backend
├── models/                  # Pre-trained ML models
├── Data/                    # Historical data & transformation pipelines
├── README.md                # Project overview
```

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 👥 Authors :
NAME : KSHITIJ KRISHNA 
GITHUB LINK : https://github.com/kshitijKrishna
LINKEDIN : https://www.linkedin.com/in/kshitij-krishna-a912a3317

NAME : VIBHANSHU VAIBHAV
GITHUB LINK : https://github.com/lifesark
LINKEDIN : https://www.linkedin.com/in/thevibhanshu/

---

## 🤝 Contributing

We welcome contributions and suggestions!

1. Fork this repository
2. Create a new branch (`feature-xyz`)
3. Commit your changes
4. Open a Pull Request

For feature requests or bug reports, open an Issue on GitHub.

---

## 📬 Contact

