import React, { useState } from "react";
import "./App.css";
import logo from "./logo.png";

const initialForm = {
  matcode: "",
  vendorid: "",
  order_value: "",
  iso_certificate: "",
  msme_status: "",
  vendor_category: "",
  tender_type: "",
  delivery_date: "",
  volume_kg: "",
};

const vendorCategories = ["Regional", "National", "International"];
const yesNo = ["Yes", "No"];
const msmeStatuses = ["Micro", "Small", "Medium", "Not MSME"];
const tenderTypes = ["Open", "Limited", "Single"];

const tabs = [
  { id: "vendor", label: "Vendor Profile", icon: "👥" },
  { id: "order", label: "Order Details", icon: "📋" }
];

function App() {
  const [form, setForm] = useState(initialForm);
  const [orderDate, setOrderDate] = useState(new Date().toISOString().split('T')[0]);
  const [result, setResult] = useState(null);
  const [resultData, setResultData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [formErrors, setFormErrors] = useState({});
  const [activeTab, setActiveTab] = useState("vendor");
  const [vendorType, setVendorType] = useState(null);
  const [predictionType, setPredictionType] = useState(null);

  // Set vendor ID to "NEW" when selecting non-existing vendor
  const handleVendorTypeSelect = (type) => {
    setVendorType(type);
    if (type === 'non-existing') {
      setForm(prev => ({ ...prev, vendorid: "NEW" }));
    }
  };

  const validateForm = () => {
    const errors = {};
    const requiredFields = ["matcode", "order_value", "iso_certificate",
                           "msme_status", "vendor_category", "tender_type"];

    // Add delivery_date as required only for on-time prediction
    if (predictionType === 'predict') {
      requiredFields.push("delivery_date");
    }

    // Add volume_kg as required only for delivery estimation
    if (predictionType === 'estimate') {
      requiredFields.push("volume_kg");
    }

    // Always require vendorId but handle it specially
    if (!form.vendorid) {
      errors.vendorid = "Please enter a Vendor ID or use 'NEW' for non-existing vendors";
    }

    requiredFields.forEach(field => {
      if (!form[field]) {
        errors[field] = "This field is required";
      }
    });

    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
    if (formErrors[name]) {
      setFormErrors(prev => ({ ...prev, [name]: null }));
    }
  };

  const handleOrderDateChange = (e) => {
    setOrderDate(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    setResult(null);
    setResultData(null);
    setLoading(true);

    try {
      // Base form data - common for both endpoints
      const formData = {
        matcode: form.matcode,
        vendorid: form.vendorid,
        order_value: Number(form.order_value),
        iso_certificate: form.iso_certificate,
        msme_status: form.msme_status,
        vendor_category: form.vendor_category,
        tender_type: form.tender_type
      };

      // Only add delivery_date for on-time prediction
      if (predictionType === 'predict') {
        formData.delivery_date = form.delivery_date;
      }

      // Only add volume_kg for delivery estimation
      if (predictionType === 'estimate') {
        formData.volume_kg = form.volume_kg ? Number(form.volume_kg) : 1000.0;
      }

      console.log('Sending request with data:', formData);
      console.log('Prediction type:', predictionType);

      // Build the endpoint URL with order_date parameter for estimation
      let endpoint = predictionType === 'predict' ? '/predict/on-time' : '/predict/delivery-date';
      if (predictionType === 'estimate' && orderDate) {
        endpoint += `?order_date=${orderDate}`;
      }

      const url = `http://localhost:8000${endpoint}`;
      console.log('Request URL:', url);

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ orders: [formData] })
      });

      console.log('Response status:', response.status);

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Error response:', errorData);
        throw new Error(errorData.detail || 'Prediction request failed');
      }

      const data = await response.json();
      console.log('Response data:', data);
      setResultData(data[0]);

      if (predictionType === 'predict') {
        setResult(`On-time Delivery Probability: ${data[0].on_time_probability}%`);
      } else {
        setResult(`Estimated Delivery Date: ${data[0].estimated_delivery_date}`);
      }
    } catch (error) {
      console.error('Error details:', error);
      setResult(`Error: ${error.message || 'Failed to get prediction. Please try again.'}`);
    } finally {
      setLoading(false);
    }
  };

  const handleNext = () => {
    if (activeTab === "vendor") setActiveTab("order");
    else if (activeTab === "order") setActiveTab("vendor");
  };

  const handlePrevious = () => {
    if (activeTab === "order") setActiveTab("vendor");
  };

  const handleBackToVendorType = () => {
    setVendorType(null);
    setActiveTab("vendor");
    setForm(initialForm);
    setResult(null);
    setResultData(null);
  };

  const renderInput = (name, label, type = "text", options = null) => (
    <label className={formErrors[name] ? "error" : ""}>
      <span>{label}</span>
      {options ? (
          <select
          name={name}
          value={form[name]}
            onChange={handleChange}
          className={formErrors[name] ? "error" : ""}
          >
          <option value="">Select {label}</option>
          {options.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
      ) : (
        <>
          <input
            type={type}
            name={name}
            value={form[name]}
            onChange={handleChange}
            className={formErrors[name] ? "error" : ""}
            placeholder={`Enter ${label.toLowerCase()}`}
            readOnly={name === "vendorid" && vendorType === "non-existing"}
          />
          {name === "vendorid" && vendorType === "non-existing" && (
            <div className="input-hint">Auto-filled for new vendor</div>
          )}
        </>
      )}
      {formErrors[name] && <span className="error-message">{formErrors[name]}</span>}
    </label>
  );

  const renderNavigationButtons = () => (
    <div className="navigation-buttons">
      {activeTab === "order" && (
        <button type="button" onClick={handlePrevious} className="nav-btn prev-btn">
          ← Previous
        </button>
      )}
      {activeTab === "order" ? (
        <button
          type="submit"
          className="nav-btn risk-btn"
          disabled={loading}
        >
          {loading ? (
            <>
              <span className="spinner"></span>
              Predicting...
            </>
          ) : (
            'Get Prediction'
          )}
        </button>
      ) : (
        <button type="button" onClick={handleNext} className="nav-btn next-btn">
          Next →
        </button>
      )}
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case "order":
        return (
          <div className="form-section">
            {renderInput("matcode", "Material Code", "text")}
            {renderInput("order_value", "Order Value", "number")}
            {/* Only show volume_kg for delivery estimation */}
            {predictionType === 'estimate' && renderInput("volume_kg", "Volume (KG)", "number")}
            {/* Show delivery_date for on-time prediction */}
            {predictionType === 'predict' && renderInput("delivery_date", "Delivery Date", "date")}
            {/* Show order_date for delivery estimation */}
            {predictionType === 'estimate' && (
              <label>
                <span>Order Date</span>
                <input
                  type="date"
                  value={orderDate}
                  onChange={handleOrderDateChange}
                  className="order-date-input"
                />
              </label>
            )}
            {renderInput("tender_type", "Tender Type", "text", tenderTypes)}
            {renderNavigationButtons()}
          </div>
        );
      case "vendor":
        return (
          <div className="form-section">
            {renderInput("vendorid", "Vendor ID", "text")}
            {renderInput("vendor_category", "Vendor Category", "text", vendorCategories)}
            {renderInput("iso_certificate", "ISO Certificate", "text", yesNo)}
            {renderInput("msme_status", "MSME Status", "text", msmeStatuses)}
            {renderNavigationButtons()}
          </div>
        );
      default:
        return null;
    }
  };

  const getRiskLevel = (prob) => {
    if (prob >= 90) return { level: "Low Risk", color: "#4CAF50" };
    if (prob >= 75) return { level: "Moderate Risk", color: "#FFC107" };
    if (prob >= 60) return { level: "Medium Risk", color: "#FF9800" };
    return { level: "High Risk", color: "#F44336" };
  };

  const renderResult = () => {
    if (!result) return null;

    // For on-time delivery prediction
    if (predictionType === 'predict' && resultData) {
      const probability = parseFloat(resultData.on_time_probability);
      const riskLevel = getRiskLevel(probability);

      return (
        <div className="result-container">
          <div className="result-box">
            <div className="result-header">
              <h3>Prediction Result</h3>
              <div className="probability-circle">
                <svg viewBox="0 0 36 36" className="circular-chart" style={{ "--progress": probability }}>
                  <path
                    className="circle-bg"
                    d="M18 2.0845
                      a 15.9155 15.9155 0 0 1 0 31.831
                      a 15.9155 15.9155 0 0 1 0 -31.831"
                  />
                  <path
                    className="circle"
                    d="M18 2.0845
                      a 15.9155 15.9155 0 0 1 0 31.831
                      a 15.9155 15.9155 0 0 1 0 -31.831"
                    transform="rotate(-90 18 18)"
                  />
                  <text x="18" y="18" className="percentage">
                    {probability}%
                  </text>
                </svg>
              </div>
              <div className="risk-level" style={{ color: riskLevel.color }}>
                {resultData.risk_level || riskLevel.level} Risk
              </div>
            </div>
            <div className="result-details">
              <div className="detail-item">
                <span className="detail-label">On-time Delivery Probability</span>
                <span className="detail-value">{probability}%</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Prediction</span>
                <span className="detail-value">{resultData.prediction}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Recommendation</span>
                <span className="detail-value">
                  {probability >= 90
                    ? "Proceed with the vendor"
                    : probability >= 75
                      ? "Proceed with caution"
                      : "Consider alternative vendors"}
                </span>
              </div>
            </div>
          </div>
        </div>
      );
    }

    // For delivery date estimation
    if (predictionType === 'estimate' && resultData) {
      return (
        <div className="result-container">
          <div className="result-box">
            <div className="result-header">
              <h3>Delivery Date Estimation</h3>
              <div className="calendar-icon">📅</div>
            </div>
            <div className="result-details">
              <div className="detail-item">
                <span className="detail-label">Estimated Delivery Date</span>
                <span className="detail-value">{resultData.estimated_delivery_date}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Order Date</span>
                <span className="detail-value">{orderDate}</span>
              </div>
              {resultData.confidence_days && (
                <>
                  <div className="detail-item">
                    <span className="detail-label">Earliest Possible Date</span>
                    <span className="detail-value">{resultData.confidence_days.earliest}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Latest Possible Date</span>
                    <span className="detail-value">{resultData.confidence_days.latest}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Margin (days)</span>
                    <span className="detail-value">{resultData.confidence_days.margin_days}</span>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      );
    }

    // Generic error result
    return (
      <div className="result-container">
        <div className="result-box error-result">
          <div className="result-header">
            <h3>Error</h3>
          </div>
          <div className="result-details">
            <div className="detail-item">
              <span className="detail-value error-message">{result}</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="main-bg">
      <header className="header-bar">
        <img src={logo} alt="BHEL Logo" className="bhel-logo" />
        <div className="header-title">
          <span className="hindi">बी.एच.ई.एल वेंडर डिलीवरी प्रेडिक्शन</span>
          <span className="english">B.H.E.L Vendor Delivery Prediction</span>
        </div>
      </header>
      <div className="app-container">
        <h2 className="subtitle">B.H.E.L Vendor Delivery Prediction Portal</h2>
        <div className="content-wrapper">
          <div className="form-section-wrapper">
            {!vendorType ? (
              <div className="vendor-type-selection">
                <button
                  className="vendor-type-btn existing"
                  onClick={() => handleVendorTypeSelect('existing')}
                >
                  <span className="btn-icon">✓</span>
                  EXISTING VENDOR
                </button>
                <button
                  className="vendor-type-btn non-existing"
                  onClick={() => handleVendorTypeSelect('non-existing')}
                >
                  <span className="btn-icon">+</span>
                  NON-EXISTING VENDOR
                </button>
              </div>
            ) : !predictionType ? (
              <div className="prediction-type-selection">
                <button
                  className="prediction-type-btn predict"
                  onClick={() => setPredictionType('predict')}
                >
                  <span className="btn-icon">📊</span>
                  Predict On Time Delivery
                </button>
                <button
                  className="prediction-type-btn estimate"
                  onClick={() => setPredictionType('estimate')}
                >
                  <span className="btn-icon">📅</span>
                  Estimate Delivery
                </button>
                <button
                  className="back-to-vendor-btn"
                  onClick={() => setVendorType(null)}
                >
                  <span className="back-icon">←</span>
                  Back to Vendor Type
                </button>
              </div>
            ) : (
              <form onSubmit={handleSubmit}>
                <div className="tabs-container">
                  {tabs.map(tab => (
                    <button
                      key={tab.id}
                      type="button"
                      className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
                      onClick={() => setActiveTab(tab.id)}
                    >
                      <span className="tab-icon">{tab.icon}</span>
                      {tab.label}
                    </button>
                  ))}
                </div>
                <div className="form-content">
                  {renderTabContent()}
                </div>
                <div className="back-button-container">
                  <button
                    type="button"
                    className="back-to-vendor-type"
                    onClick={() => {
                      setPredictionType(null);
                      setActiveTab("vendor");
                      setForm(initialForm);
                      setResult(null);
                      setResultData(null);
                    }}
                  >
                    <span className="back-icon">←</span>
                    Select Prediction Type
                  </button>
                </div>
              </form>
            )}
          </div>
          {result && (
            <div className="result-section">
              {renderResult()}
            </div>
          )}
        </div>
      </div>
      <footer className="footer-bar">
        <div className="footer-content">
          <div className="footer-left">
            <img src={logo} alt="BHEL Logo" className="footer-logo" />
            <div className="footer-text">
              <div className="footer-title">BHEL Haridwar</div>
              <div className="footer-subtitle">Vendor Risk Prediction Portal</div>
            </div>
          </div>
          <div className="footer-right">
            <a href="#" className="footer-link">About Us</a>
            <a href="#" className="footer-link">Contact</a>
            <a href="#" className="footer-link">Help</a>
          </div>
        </div>
        <div className="footer-copyright">
          © {new Date().getFullYear()} BHEL Haridwar. All rights reserved.
        </div>
      </footer>
    </div>
  );
}

export default App;