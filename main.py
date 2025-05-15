"""
CLI entrypoint for multi-modal ADRD diagnosis (interactive version).
"""
import sys
from router import diagnose_adrd

def prompt_for_input():
    print("Welcome to the Multi-modal ADRD Diagnosis System.\n")
    print("Please enter the information you have. Leave blank if not applicable.\n")
    # Prompt for all possible fields
    symptom_score = input("Symptom score (clinical, integer): ")
    test_score = input("Cognitive test score (integer): ")
    biomarker_level = input("Biomarker level (float): ")
    spectrum_path = input("Raman spectrum file path: ")
    image_path = input("MRI image path: ")
    # Build input_data dict
    input_data = {}
    if symptom_score.strip():
        input_data["symptom_score"] = int(symptom_score)
    if test_score.strip():
        input_data["test_score"] = int(test_score)
    if biomarker_level.strip():
        input_data["biomarker_level"] = float(biomarker_level)
    if spectrum_path.strip():
        input_data["spectrum_path"] = spectrum_path.strip()
    if image_path.strip():
        input_data["image_path"] = image_path.strip()
    return input_data

def main():
    input_data = prompt_for_input()
    # Ask user which technique to use
    print("\nWhich technique would you like to use?")
    print("1. Clinical evaluation")
    print("2. Cognitive testing")
    print("3. Laboratory test")
    print("4. Raman spectroscopy test")
    print("5. Imaging (MRI)")
    technique_map = {
        "1": "clinical",
        "2": "cognitive",
        "3": "lab",
        "4": "raman",
        "5": "imaging"
    }
    selected = input("Enter the number of the technique to use: ").strip()
    technique = technique_map.get(selected)
    if not technique:
        print("Invalid selection. Exiting.")
        sys.exit(1)
    print(f"\nSelected technique: {technique.capitalize()} evaluation. Starting...")
    # Run the selected technique
    from diagnostics.clinical import clinical_evaluation
    from diagnostics.cognitive import cognitive_testing
    from diagnostics.lab import laboratory_test
    from diagnostics.raman import raman_spectroscopy_test
    from diagnostics.imaging import imaging_ad_prediction
    if technique == "clinical":
        ad_detected = clinical_evaluation(input_data)
    elif technique == "cognitive":
        ad_detected = cognitive_testing(input_data)
    elif technique == "lab":
        ad_detected = laboratory_test(input_data)
    elif technique == "raman":
        ad_detected = raman_spectroscopy_test(input_data)
    elif technique == "imaging":
        ad_detected = imaging_ad_prediction(input_data)
    else:
        print("Unknown technique. Exiting.")
        sys.exit(1)
    if not ad_detected:
        print("\nAD detected: False")
        return
    print("\nAD detected: True.\nNow starting plasma p-tau217 blood test for confirmation...")
    # Prompt for ptau217_level
    ptau217_level = input("Enter plasma p-tau217 level (pg/mL): ")
    try:
        ptau217_level = float(ptau217_level)
    except Exception:
        print("Invalid p-tau217 value. Exiting.")
        sys.exit(1)
    input_data["ptau217_level"] = ptau217_level
    from diagnostics.lab import ptau217_blood_test
    ptau217_result = ptau217_blood_test(input_data)
    print(f"\nPlasma p-tau217 blood test result: {'Positive' if ptau217_result else 'Negative'}")
    print(f"\nFinal AD diagnosis: {ptau217_result}")

if __name__ == "__main__":
    main()
