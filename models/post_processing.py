# Post-processing functions for supply chain predictions

def enhanced_probability_adjustment(probability, lead_time=30, vendor_category='National',
                                    is_new_vendor=0, iso_certified='Yes', msme_status='Medium',
                                    blacklisted=False, on_hold=False, **kwargs):
    """
    Adjust the raw model probability based on business rules and risk factors.

    Balanced approach that:
    1. Reduces the extreme impact of ISO certification
    2. Increases the impact of MSME status
    3. Makes lead time impact more linear
    4. Considers combinations of risk factors more holistically

    Parameters:
    -----------
    probability : float
        Raw probability from the model (0.0 to 1.0)
    lead_time : int
        Lead time in days
    vendor_category : str
        'Regional', 'National', or 'International'
    is_new_vendor : int
        1 if new vendor, 0 otherwise
    iso_certified : str
        'Yes' or 'No'
    msme_status : str
        'Micro', 'Small', 'Medium', 'Large', or 'Not MSME'
    blacklisted : bool
        True if vendor is blacklisted
    on_hold : bool
        True if vendor is on hold
    **kwargs : dict
        Additional parameters (for future extensibility)

    Returns:
    --------
    float
        Adjusted probability value (0.0 to 1.0)
    """
    # Make a copy of the original probability for adjustment
    adjusted_prob = probability

    # ---- Hard blockers (overriding factors) ----
    # Blacklisted vendors get minimal chances
    if blacklisted:
        return max(probability * 0.3, 0.05)

    # Vendors on hold get reduced chances
    if on_hold:
        return max(probability * 0.5, 0.1)

    # ---- Lead Time Impact (more linear approach) ----
    if lead_time < 7:  # Ultra-short lead time
        adjusted_prob *= 0.85
    elif lead_time < 15:  # Short lead time
        adjusted_prob *= 0.92
    elif lead_time > 60:  # Extended lead time
        adjusted_prob *= 1.08  # Bonus for generous lead time
        adjusted_prob = min(adjusted_prob, 0.95)  # Cap at 95%

    # ---- Vendor Category Impact ----
    if vendor_category == 'International':
        if lead_time < 30:  # Short lead time for international
            adjusted_prob *= 0.92  # Slight penalty
        else:
            # No adjustment for adequate lead time
            pass
    elif vendor_category == 'Regional':
        adjusted_prob *= 1.05  # Slight bonus for local vendors
        adjusted_prob = min(adjusted_prob, 0.95)  # Cap at 95%

    # ---- MSME Status Impact (significantly increased) ----
    if msme_status == 'Micro':
        adjusted_prob *= 0.85  # 15% reduction (was only ~1% before)
    elif msme_status == 'Small':
        adjusted_prob *= 0.92  # 8% reduction 
    elif msme_status == 'Medium':
        # No adjustment for Medium (baseline)
        pass
    elif msme_status == 'Large' or msme_status == 'Not MSME':
        adjusted_prob *= 1.05  # 5% bonus for larger organizations
        adjusted_prob = min(adjusted_prob, 0.96)  # Cap at 96%

    # ---- ISO Certification Impact (reduced from original model) ----
    if iso_certified != 'Yes':
        if is_new_vendor == 1:
            # Reduced impact for non-ISO new vendors (was ~0.4 before)
            adjusted_prob *= 0.70  # Now 30% reduction instead of 60%
        else:
            # Reduced impact for non-ISO existing vendors
            adjusted_prob *= 0.85  # 15% reduction

    # ---- Risk Combinations (more balanced approach) ----
    # High risk combination (reduced harshness)
    if lead_time < 7 and is_new_vendor == 1 and iso_certified != 'Yes':
        adjusted_prob = min(adjusted_prob, 0.35)  # Cap of 35% (more realistic)

    # Very high risk combination
    if lead_time < 7 and is_new_vendor == 1 and iso_certified != 'Yes' and msme_status == 'Micro':
        adjusted_prob = min(adjusted_prob, 0.25)  # Cap of 25%

    # Low risk combination
    if lead_time > 45 and is_new_vendor == 0 and iso_certified == 'Yes':
        adjusted_prob = max(adjusted_prob, 0.70)  # Floor of 70% for low risk

    # ---- Final caps and floors ----
    # Ensure probability stays within valid range
    return max(0.01, min(adjusted_prob, 0.99))


def identify_risk_factors(input_data, prediction, probability):
    """
    Identify key risk factors for a given prediction.

    Parameters:
    -----------
    input_data : dict or pandas.Series
        Input data used for prediction
    prediction : str
        Prediction result ("On Time" or "Delayed")
    probability : float
        Probability value (0.0 to 1.0)

    Returns:
    --------
    dict
        Dictionary with risk factors and their severities
    """
    risk_factors = {}

    # Only analyze risk factors for delayed or borderline predictions
    if prediction == "Delayed" or probability < 0.7:
        # Check lead time
        lead_time = input_data.get('LeadTimeDays', 0)
        if lead_time < 7:
            risk_factors["Insufficient Lead Time"] = "High"
        elif lead_time < 15:
            risk_factors["Short Lead Time"] = "Medium"

        # Check vendor status
        if input_data.get('IsNewVendor', 0) == 1:
            risk_factors["New Vendor"] = "Medium"

        # Check ISO certification
        if input_data.get('ISO9001_Certified', '') != 'Yes':
            if input_data.get('IsNewVendor', 0) == 1:
                risk_factors["Non-ISO Certified New Vendor"] = "High"
            else:
                risk_factors["Non-ISO Certified Vendor"] = "Medium"

        # Check MSME status
        if input_data.get('MSME_Status', '') == 'Micro':
            risk_factors["Micro MSME Vendor"] = "Medium"  # Upgraded from Low
        elif input_data.get('MSME_Status', '') == 'Small':
            risk_factors["Small MSME Vendor"] = "Low"

        # Check vendor category
        if input_data.get('VendorCategory', '') == 'International' and lead_time < 30:
            risk_factors["International Vendor with Short Lead Time"] = "Medium"

        # Check tender type
        if input_data.get('TenderType', '') == 'Open':
            risk_factors["Open Tender"] = "Low"

        # Check seasonal factors
        if input_data.get('IsMonsoonSeason', 0) == 1:
            risk_factors["Monsoon Season"] = "Low"

        if input_data.get('IsQuarterEnd', 0) == 1:
            risk_factors["Quarter End Pressure"] = "Low"

        # Check for blockers
        if input_data.get('BlacklistStatus', 'No') == 'Yes':
            risk_factors["Blacklisted Vendor"] = "Critical"

        if input_data.get('HoldStatus', 'No') == 'Yes':
            risk_factors["Vendor On Hold"] = "High"

    return risk_factors