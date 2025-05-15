"""
Central router for multi-modal ADRD diagnosis.
"""
from diagnostics.clinical import clinical_evaluation
from diagnostics.cognitive import cognitive_testing
from diagnostics.lab import laboratory_test, ptau217_blood_test
from diagnostics.raman import raman_spectroscopy_test
from diagnostics.imaging import imaging_ad_prediction

def diagnose_adrd(input_data: dict) -> bool:
    """
    Central ADRD diagnosis algorithm. Determines the most suitable diagnostic technique
    based on input_data, calls the technique, and returns the result.
    Args:
        input_data (dict): User input. If 'type' is missing, infers the technique.
    Returns:
        bool: True if AD detected, False otherwise.
    """
    dtype = input_data.get("type")
    if not dtype:
        # Auto-infer technique based on input keys
        if "symptom_score" in input_data:
            dtype = "clinical"
        elif "test_score" in input_data:
            dtype = "cognitive"
        elif "biomarker_level" in input_data:
            dtype = "lab"
        elif "spectrum_path" in input_data:
            dtype = "raman"
        elif "image_path" in input_data:
            dtype = "imaging"
        else:
            raise ValueError("Cannot infer diagnostic type from input data.")
    # Run the first stage (multi-modal diagnosis)
    if dtype == "clinical":
        ad_detected = clinical_evaluation(input_data)
    elif dtype == "cognitive":
        ad_detected = cognitive_testing(input_data)
    elif dtype == "lab":
        ad_detected = laboratory_test(input_data)
    elif dtype == "raman":
        ad_detected = raman_spectroscopy_test(input_data)
    elif dtype == "imaging":
        ad_detected = imaging_ad_prediction(input_data)
    else:
        raise ValueError(f"Unknown diagnostic type: {dtype}")

    # If AD detected (even very mild), run p-tau217 blood test for confirmation
    if ad_detected:
        # Only run if ptau217_level is provided
        if "ptau217_level" in input_data:
            ptau217_result = ptau217_blood_test(input_data)
            # Only confirm AD if both are positive
            return ptau217_result
        # If no ptau217_level, return AD detected from first stage
        return ad_detected
    # NON-AD case
    return ad_detected
