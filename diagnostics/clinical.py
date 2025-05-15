"""
Clinical evaluation stub for ADRD diagnosis.
"""

def clinical_evaluation(data: dict) -> bool:
    """
    Perform clinical evaluation for ADRD diagnosis.
    Args:
        data (dict): Clinical data, e.g., symptom scores.
    Returns:
        bool: True if AD detected, False otherwise.
    """
    # Stub logic: AD if symptom_score >= 7
    return data.get("symptom_score", 0) >= 7
