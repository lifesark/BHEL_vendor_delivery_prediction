from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timedelta
import uvicorn

from models.post_processing import enhanced_probability_adjustment, identify_risk_factors

# Initialize FastAPI app
app = FastAPI(title="Supply Chain Delivery Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define base order input model
class BaseOrderInput(BaseModel):
    matcode: str = Field(..., description="Material code")
    vendorid: str = Field(..., description="Vendor ID (use 'NEW' for new vendors)")
    order_value: float = Field(..., description="Order value")
    iso_certificate: str = Field(..., description="ISO certification (Yes/No)")
    msme_status: str = Field(..., description="MSME status")
    vendor_category: str = Field(..., description="Regional/National/International")
    tender_type: str = Field(..., description="Single/Limited/Open")
    volume_kg: Optional[float] = Field(None, description="Volume in KG")


# For on-time prediction, we need delivery_date
class OnTimeOrderInput(BaseOrderInput):
    delivery_date: str = Field(..., description="Target delivery date (YYYY-MM-DD)")


# For delivery date estimation, we don't want delivery_date
class DeliveryEstimateOrderInput(BaseOrderInput):
    pass  # No delivery_date field for this endpoint


class OnTimeOrderInputList(BaseModel):
    orders: List[OnTimeOrderInput]


class DeliveryEstimateOrderInputList(BaseModel):
    orders: List[DeliveryEstimateOrderInput]


class OnTimePredictionResponse(BaseModel):
    order_id: int
    on_time_probability: float
    prediction: str
    risk_level: str


class DeliveryEstimateResponse(BaseModel):
    order_id: int
    estimated_delivery_date: str
    confidence_days: Dict[str, str]


# Global variables to store models and metadata
classifier = None
lead_time_estimator = None
material_models = None
metadata = None
post_processing_functions = None
required_features = None
required_reg_features = None


# Load models and mapping functions
def load_models():
    global classifier, lead_time_estimator, material_models, metadata, post_processing_functions, required_features, required_reg_features

    if classifier is None:
        # Load the classifier and metadata
        classifier = joblib.load('models/ontime_delivery_classifier.pkl')
        lead_time_estimator = joblib.load('models/lead_time_estimator.pkl')

        # Try to load material-specific models
        try:
            material_models = joblib.load('models/material_specific_models.pkl')
        except:
            material_models = None

        # Load metadata
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)

        # Import post-processing functions
        post_processing_functions = {
            'adjust_probability': enhanced_probability_adjustment,
            'identify_risks': identify_risk_factors
        }

        # Extract required features from model
        if isinstance(classifier, object) and hasattr(classifier,
                                                      'named_steps') and 'preprocessor' in classifier.named_steps:
            preprocessor = classifier.named_steps['preprocessor']
            if hasattr(preprocessor, 'transformers_'):
                required_features = []
                for _, _, cols in preprocessor.transformers_:
                    if cols is not None:
                        required_features.extend(cols)

        # Get required features for regression from metadata
        required_reg_features = metadata.get('regression_features', [])

    return {
        'classifier': classifier,
        'estimator': lead_time_estimator,
        'material_models': material_models,
        'metadata': metadata,
        'adjust_probability': post_processing_functions['adjust_probability'] if post_processing_functions else None,
        'identify_risks': post_processing_functions['identify_risks'] if post_processing_functions else None,
        'required_features': required_features,
        'required_reg_features': required_reg_features
    }


# Transform client input to model input format with all required columns
def transform_input(client_input: BaseOrderInput, order_date: str = None, models_dict=None,
                    for_estimation=False) -> pd.DataFrame:
    """Convert client input format to model input format with all required columns"""

    if models_dict is None:
        models_dict = load_models()

    # Convert the current date if order_date is not provided
    if not order_date:
        order_date = datetime.now().strftime('%Y-%m-%d')

    # Calculate lead time if delivery date is provided (for OnTimeOrderInput)
    lead_time_days = None
    if hasattr(client_input, 'delivery_date') and client_input.delivery_date:
        try:
            delivery_date = datetime.strptime(client_input.delivery_date, '%Y-%m-%d')
            order_datetime = datetime.strptime(order_date, '%Y-%m-%d')
            lead_time_days = (delivery_date - order_datetime).days
        except:
            lead_time_days = None

    # Check if new vendor
    is_new_vendor = 1 if client_input.vendorid.upper() in ['NEW', 'NEW_VENDOR'] else 0

    # Get order month
    current_month = datetime.strptime(order_date, '%Y-%m-%d').month
    is_monsoon = 1 if current_month in [6, 7, 8] else 0
    is_quarter_end = 1 if current_month in [3, 6, 9, 12] else 0

    # Base input values from client input
    base_input = {
        'MaterialCode': client_input.matcode,
        'VendorID': client_input.vendorid if not is_new_vendor else 'NEW_VENDOR',
        'PO_Amount': client_input.order_value,
        'ISO9001_Certified': client_input.iso_certificate,
        'MSME_Status': client_input.msme_status,
        'VendorCategory': client_input.vendor_category,
        'TenderType': client_input.tender_type,
        'IsNewVendor': is_new_vendor,
        'MaterialType': 'Raw',  # Default if not provided
        'LeadTimeDays': lead_time_days if lead_time_days and lead_time_days > 0 else 30,  # Default to 30 days
        'OrderMonth': current_month,
        'IsMonsoonSeason': is_monsoon,
        'IsQuarterEnd': is_quarter_end,
        'PriorityFlag': 'Medium',
        'InspectionRequired': 'Yes',
        'PenaltyClause': 'Yes',
        'BlacklistStatus': 'No',
        'HoldStatus': 'No',
        'PastDelayCount': 0,
        'PastRejectionRate': 0.0,
        'PaymentTerms': 30,
        'VendorReliabilityScore': 70.0 if is_new_vendor == 0 else 50.0,
        'MaterialComplexityScore': 50.0,
        'VolumeKG': client_input.volume_kg if client_input.volume_kg else 1000.0,  # Default 1000 KG
        'LogisticsDistanceKM': 500.0  # Default logistics distance
    }

    # Add derived features
    base_input['LeadTime_Category'] = get_lead_time_category(base_input['LeadTimeDays'])
    base_input['HighRiskCombo'] = 1 if (base_input['LeadTimeDays'] < 7 and
                                        base_input['IsNewVendor'] == 1 and
                                        base_input['ISO9001_Certified'] == 'No') else 0
    base_input['LeadTime_x_VendorReliability'] = base_input['LeadTimeDays'] * base_input['VendorReliabilityScore']
    base_input['LeadTime_x_MaterialComplexity'] = base_input['LeadTimeDays'] * base_input['MaterialComplexityScore']
    base_input['Monsoon_MaterialComplexity'] = base_input['IsMonsoonSeason'] * base_input['MaterialComplexityScore']

    # Create DataFrame with all required columns
    input_df = pd.DataFrame([base_input])

    # Ensure all required features are present
    if for_estimation and models_dict['required_reg_features']:
        for feature in models_dict['required_reg_features']:
            if feature not in input_df.columns:
                input_df[feature] = get_default_value(feature)
    elif not for_estimation and models_dict['required_features']:
        for feature in models_dict['required_features']:
            if feature not in input_df.columns:
                input_df[feature] = get_default_value(feature)

    return input_df



def get_lead_time_category(lead_time):
    """Convert lead time days to category"""
    if lead_time < 8:
        return 'Ultra-short (1-7d)'
    elif lead_time < 16:
        return 'Short (8-15d)'
    elif lead_time < 46:
        return 'Normal (16-45d)'
    elif lead_time < 91:
        return 'Long (46-90d)'
    else:
        return 'Very Long (91+d)'


def get_default_value(feature_name):
    """Return default values for features based on feature name"""
    defaults = {
        'VendorReliabilityScore': 60.0,
        'MaterialComplexityScore': 50.0,
        'PastDelayCount': 0,
        'PastRejectionRate': 0.0,
        'IsMonsoonSeason': 0,
        'IsQuarterEnd': 0,
        'PenaltyClause': 'No',
        'BlacklistStatus': 'No',
        'HoldStatus': 'No',
        'PaymentTerms': 30,
        'LeadTime_Category': 'Normal (16-45d)',
        'HighRiskCombo': 0,
        'LeadTime_x_VendorReliability': 1800,  # 30 * 60
        'LeadTime_x_MaterialComplexity': 1500,  # 30 * 50
        'Monsoon_MaterialComplexity': 0,
        'VolumeKG': 1000.0,
        'LogisticsDistanceKM': 500.0
    }

    # Return the default value or 0 if not found
    return defaults.get(feature_name, 0)


@app.post("/predict/on-time", response_model=List[OnTimePredictionResponse])
def predict_ontime_delivery(orders: OnTimeOrderInputList):
    """Predict on-time probability for orders with delivery dates"""
    # Load models
    models = load_models()

    predictions = []

    for i, order in enumerate(orders.orders):
        try:
            # Transform to model input with all required columns
            input_df = transform_input(order, models_dict=models)

            if input_df['LeadTimeDays'].iloc[0] <= 0:
                predictions.append(OnTimePredictionResponse(
                    order_id=i,
                    on_time_probability=0,
                    prediction="Error: Invalid delivery date",
                    risk_level="Unknown"
                ))
                continue

            # Get prediction
            raw_probability = models['classifier'].predict_proba(input_df)[0][1]

            # Apply post-processing with enhanced function
            adjusted_probability = models['adjust_probability'](
                probability=raw_probability,
                lead_time=input_df['LeadTimeDays'].iloc[0],
                vendor_category=input_df['VendorCategory'].iloc[0],
                is_new_vendor=input_df['IsNewVendor'].iloc[0],
                iso_certified=input_df['ISO9001_Certified'].iloc[0],
                msme_status=input_df['MSME_Status'].iloc[0],
                blacklisted=(input_df['BlacklistStatus'].iloc[0] == 'Yes'),
                on_hold=(input_df['HoldStatus'].iloc[0] == 'Yes')
            )

            # Determine risk level
            risk_level = "Low"
            if adjusted_probability < 0.4:
                risk_level = "High"
            elif adjusted_probability < 0.6:
                risk_level = "Medium"

            predictions.append(OnTimePredictionResponse(
                order_id=i,
                on_time_probability=round(adjusted_probability * 100, 1),
                prediction="On Time" if adjusted_probability >= 0.5 else "Delayed",
                risk_level=risk_level
            ))
        except Exception as e:
            predictions.append(OnTimePredictionResponse(
                order_id=i,
                on_time_probability=0,
                prediction=f"Error: {str(e)}",
                risk_level="Unknown"
            ))

    return predictions


@app.post("/predict/delivery-date", response_model=List[DeliveryEstimateResponse])
def estimate_delivery_date(orders: DeliveryEstimateOrderInputList, order_date: Optional[str] = Query(None, description="Order date (YYYY-MM-DD format). Defaults to today's date if not provided.")):
    """Estimate delivery date for new orders based on order characteristics"""
    # Use current date if not provided
    if not order_date:
        order_date = datetime.now().strftime('%Y-%m-%d')

    # Load models
    models = load_models()

    predictions = []

    for i, order in enumerate(orders.orders):
        try:
            # Transform to model input - for_estimation=True to include regression features
            input_df = transform_input(order, order_date, models, for_estimation=True)

            # Get delivery estimate
            lead_time = 30  # Default

            if models['material_models'] and 'MaterialType' in input_df.columns:
                material_type = input_df['MaterialType'].iloc[0]
                if material_type in models['material_models']:
                    try:
                        lead_time = models['material_models'][material_type].predict(input_df)[0]
                    except Exception as model_error:
                        # Fallback to general model
                        try:
                            lead_time = models['estimator'].predict(input_df)[0]
                        except Exception as fallback_error:
                            print(f"Material model error: {model_error}, Fallback error: {fallback_error}")
                            # Use default lead time
                            lead_time = 30
                else:
                    lead_time = models['estimator'].predict(input_df)[0]
            else:
                lead_time = models['estimator'].predict(input_df)[0]

            # Calculate delivery date
            order_datetime = datetime.strptime(order_date, '%Y-%m-%d')
            delivery_date = order_datetime + timedelta(days=int(lead_time))

            # Calculate confidence interval (±20%)
            margin = int(lead_time * 0.2)
            earliest = order_datetime + timedelta(days=int(lead_time - margin))
            latest = order_datetime + timedelta(days=int(lead_time + margin))

            predictions.append(DeliveryEstimateResponse(
                order_id=i,
                estimated_delivery_date=delivery_date.strftime('%Y-%m-%d'),
                confidence_days={
                    "earliest": earliest.strftime('%Y-%m-%d'),
                    "latest": latest.strftime('%Y-%m-%d'),
                    "margin_days": str(margin)
                }
            ))
        except Exception as e:
            import traceback
            print(f"Error in delivery estimation: {str(e)}")
            print(traceback.format_exc())
            predictions.append(DeliveryEstimateResponse(
                order_id=i,
                estimated_delivery_date="Error",
                confidence_days={
                    "error": str(e)
                }
            ))

    return predictions


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "2.1.0"  # Added version to help track which version is deployed
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)