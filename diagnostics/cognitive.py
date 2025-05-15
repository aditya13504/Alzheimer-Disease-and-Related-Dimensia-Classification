"""
Cognitive testing stub for ADRD diagnosis.
"""

def cognitive_testing(data: dict) -> bool:
    """
    Perform cognitive testing for ADRD diagnosis.
    Args:
        data (dict): Cognitive test data, e.g., test scores.
    Returns:
        bool: True if AD detected, False otherwise.
    """
    # Stub logic: AD if test_score <= 12
    return data.get("test_score", 30) <= 12
