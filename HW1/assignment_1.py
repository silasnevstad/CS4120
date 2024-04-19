import pandas as pd
import re
from collections import Counter
import string


# ===== Function Definitions =====


def load_and_clean_symptoms_data(file_path):
    """
    Load the symptom data from an Excel file and clean it.
    Returns a dictionary of combined symptoms and their codes.
    """
    try:
        symptoms_df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        exit()

    # Remove the first four rows of the dataframe (they are not needed)
    symptoms_df = symptoms_df.iloc[4:, :3]
    # Update the column names
    symptoms_df.columns = ['code', 'symptom', 'other_names']

    # Create a dictionary of symptoms and their codes
    symptoms_dict = dict(zip(symptoms_df['symptom'], symptoms_df['code']))

    # Create a dictionary of other names and their codes
    other_names_dict = dict(zip(symptoms_df['other_names'], symptoms_df['code']))

    # Clean the dictionaries by removing entries with non-integer code values
    symptoms_dict = {k: v for k, v in symptoms_dict.items() if isinstance(v, int)}
    other_names_dict = {k: v for k, v in other_names_dict.items() if isinstance(v, int) and pd.notna(k)}

    # Combine the two dictionaries
    combined_symptoms_dict = {**symptoms_dict, **other_names_dict}

    # Trim and lower the keys and sort the dictionary alphabetically
    combined_symptoms_dict = {k.strip(): v for k, v in combined_symptoms_dict.items()}
    combined_symptoms_dict = {k.lower(): v for k, v in combined_symptoms_dict.items()}
    combined_symptoms_dict = {k: v for k, v in sorted(combined_symptoms_dict.items(), key=lambda item: item[0])}

    return combined_symptoms_dict


def extract_symptoms_to_codes(notes, symptom_codes):
    """
    Extracts and maps symptoms to their SNOMED codes from clinical notes.
    """
    extracted_symptoms = {}
    for symptom, code in symptom_codes.items():
        # Check if the symptom or its other name is mentioned in the notes
        if re.search(r'\b' + re.escape(symptom) + r'\b', notes, re.IGNORECASE):
            extracted_symptoms[symptom] = code
    return extracted_symptoms


def process_clinical_notes(file_paths, symptom_codes):
    """
    Process clinical notes to extract symptom codes.
    """
    clinical_notes_symptoms = {}
    for file_path in file_paths:
        with open("test_set/" + file_path, 'r') as file:
            clinical_note = file.read()
        symptoms_codes = extract_symptoms_to_codes(clinical_note, symptom_codes)
        clinical_notes_symptoms[file_path] = list(symptoms_codes.values())
    return clinical_notes_symptoms


def keyword_frequency_analysis(notes):
    """
    Performs keyword frequency analysis on given notes.
    """
    # Remove punctuation and convert to lowercase
    notes_clean = notes.translate(str.maketrans('', '', string.punctuation)).lower()

    # Split the notes into words
    words = notes_clean.split()

    # Count the frequency of each word
    word_freq = Counter(words)

    # Sort the words by their frequency in descending order and return the top three
    most_common_words = word_freq.most_common(3)
    return most_common_words


def compare_symptoms(notes_symptoms_dict):
    """
    Compare symptoms between different patient notes.
    """
    comparison_results = {}
    file_names = list(notes_symptoms_dict.keys())

    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            file1, file2 = file_names[i], file_names[j]
            symptoms_file1 = set(notes_symptoms_dict[file1])
            symptoms_file2 = set(notes_symptoms_dict[file2])

            # Common symptoms
            common_symptoms = symptoms_file1.intersection(symptoms_file2)

            # Unique symptoms to each file
            unique_file1 = symptoms_file1 - common_symptoms
            unique_file2 = symptoms_file2 - common_symptoms

            comparison_results[(file1, file2)] = {
                'common_symptoms': list(common_symptoms),
                'unique_to_file1': list(unique_file1),
                'unique_to_file2': list(unique_file2)
            }

    return comparison_results


def unique_symptoms_across_notes(notes_symptoms_dict):
    """
    Find all unique symptoms across different clinical notes.
    """
    all_symptoms = set()
    for symptoms in notes_symptoms_dict.values():
        all_symptoms.update(symptoms)
    return list(all_symptoms)


def extract_patient_info(note):
    """
    Extracts patient information from clinical notes.
    """
    mrn = re.search(r'MRN: (\d+)', note)
    age = re.search(r'(\d+)-year-old', note)
    gender = re.search(r'\b(male|female)\b', note, re.IGNORECASE)

    mrn = mrn.group(1) if mrn else 'Unknown'
    age = int(age.group(1)) if age else None
    gender = gender.group(1).lower() if gender else 'unknown'

    return mrn, age, gender


def make_patient_symptom_tracking_dict(file_paths, symptom_codes):
    """
    Processes patient information from clinical notes to track patient symptoms.
    """
    patient_symptoms_dict = {}
    for file_path in file_paths:
        with open('test_set/' + file_path, 'r') as file:
            clinical_note = file.read()
        mrn, _, _ = extract_patient_info(clinical_note)
        symptoms = extract_symptoms_to_codes(clinical_note, symptom_codes)
        if mrn in patient_symptoms_dict:
            patient_symptoms_dict[mrn].extend(list(symptoms.values()))
        else:
            patient_symptoms_dict[mrn] = list(symptoms.values())

    return patient_symptoms_dict


def make_patient_records(file_paths, symptom_codes):
    """
    Processes patient information from clinical notes to create patient records.
    """
    patient_records = {}
    for file_path in file_paths:
        with open("test_set/" + file_path, 'r') as file:
            clinical_note = file.read()
        mrn, age, gender = extract_patient_info(clinical_note)
        symptoms = extract_symptoms_to_codes(clinical_note, symptom_codes)
        patient_records[mrn] = {'gender': gender, 'age': age, 'symptoms': list(symptoms.values()), 'mrn': mrn}
    return patient_records


clinical_notes_file_paths = [
    '32085846.en.txt',
    '32119083.en.txt',
    '32165205.en.txt',
    '32168162.en.txt',
    '32277408.en.txt',
]

# Part 1:
# Symptom Extraction and Manipulation
symptoms_dict = load_and_clean_symptoms_data('PHVS_SignsSymptoms_COVID-19_V2.xlsx')
clinical_notes_symptoms_dict = process_clinical_notes(clinical_notes_file_paths, symptoms_dict)
print(f"Symptom Extraction and Manipulation: {clinical_notes_symptoms_dict}")

# Keyword Frequency Analysis
clinical_notes_freq_dict = {}

# Loop through the list of clinical notes and perform keyword frequency analysis
for file_path in clinical_notes_file_paths:
    with open('test_set/' + file_path, 'r') as file:
        clinical_note = file.read()
    top_three_words = keyword_frequency_analysis(clinical_note)
    clinical_notes_freq_dict[file_path] = top_three_words

print(f"Keyword Frequency Analysis: {clinical_notes_freq_dict}")

# Part 2:
# Symptom Comparison
symptom_comparison_results = compare_symptoms(clinical_notes_symptoms_dict)
print(f"Symptom Comparison: {symptom_comparison_results}")

# Identifying Unique Symptoms in a Dataset
unique_symptoms = unique_symptoms_across_notes(clinical_notes_symptoms_dict)
print(f"Identifying Unique Symptoms in a Dataset: {unique_symptoms}")

# Part 3:
# Patient Symptom Tracking
patient_symptom_dict = make_patient_symptom_tracking_dict(clinical_notes_file_paths, symptoms_dict)
print(f"Patient Symptom Tracking: {patient_symptom_dict}")

# Nested Dictionaries for Patient Records
patient_records_dict = make_patient_records(clinical_notes_file_paths, symptoms_dict)
print(f"Nested Dictionaries for Patient Records: {patient_records_dict}")


# Add a new patient record
def add_patient_record(records, mrn, gender, age, symptoms):
    if mrn not in records:
        records[mrn] = {'gender': gender, 'age': age, 'symptoms': symptoms}
    else:
        print(f"Record for MRN {mrn} already exists.")


# Update an existing patient record
def update_patient_record(records, mrn, gender=None, age=None, symptoms=None):
    if mrn in records:
        if gender:
            records[mrn]['gender'] = gender
        if age:
            records[mrn]['age'] = age
        if symptoms:
            records[mrn]['symptoms'] = symptoms
    else:
        print(f"No record found for MRN {mrn}.")


# Delete a patient record
def delete_patient_record(records, mrn):
    if mrn in records:
        del records[mrn]
    else:
        print(f"No record found for MRN {mrn}.")


# Example usage
add_patient_record(patient_records_dict, '12345', 'male', 40, [111, 222])
update_patient_record(patient_records_dict, '12345', age=41)
delete_patient_record(patient_records_dict, '12345')


# Unit Tests
def test_extract_symptoms_to_codes():
    # Test the extract_symptoms_to_codes function with a sample input
    test_note = "Patient shows signs of fever and cough."
    test_result = extract_symptoms_to_codes(test_note, symptoms_dict)
    assert isinstance(test_result, dict), "Result should be a dictionary"


def test_keyword_frequency_analysis():
    # Test the keyword_frequency_analysis function with a sample input
    test_note = "fever fever cough"
    test_result = keyword_frequency_analysis(test_note)
    assert test_result[0][0] == 'fever' and test_result[0][1] == 2, "Top word should be 'fever' with frequency 2"


# Run unit tests
test_extract_symptoms_to_codes()
test_keyword_frequency_analysis()
