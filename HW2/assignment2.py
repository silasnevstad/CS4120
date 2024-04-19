import re

# Part I

def is_valid_rna(sequence):
    """
    Checks if a given string is a valid RNA sequence.
    
    Args:
    sequence (str): The RNA sequence to validate.
    
    Returns:
    bool: True if the sequence is valid (contains only A, C, G, U), False otherwise.
    """
    return bool(re.fullmatch(r'[ACGU]+', sequence))

def nucleotide_count(sequence):
    """
    Counts the occurrence of each nucleotide in an RNA sequence.
    
    Args:
    sequence (str): The RNA sequence to analyze.
    
    Returns:
    dict: A dictionary with nucleotides as keys and their counts as values.
    """
    return {nucleotide: sequence.count(nucleotide) for nucleotide in 'ACGU'}

def find_motifs(sequence, motif):
    """
    Identifies all occurrences of a given motif within the RNA sequence.
    
    Args:
    sequence (str): The RNA sequence to search.
    motif (str): The motif to find.
    
    Returns:
    list: A list of positions where the motif is found.
    """
    return [match.start() for match in re.finditer(r'(?={})'.format(motif), sequence)]

def complementary_sequence(sequence):
    """
    Generates the complementary sequence of a given RNA sequence.
    
    Args:
    sequence (str): The RNA sequence to complement.
    
    Returns:
    str: The complementary RNA sequence.
    """
    complement = {
        'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C',
        'R': 'R', 'Y': 'Y', 'S': 'S', 'W': 'W',
        'K': 'K', 'M': 'M', 'B': 'B', 'D': 'D',
        'H': 'H', 'V': 'V', 'N': 'N'
    }
    return ''.join(complement.get(nucleotide, 'N/A') for nucleotide in sequence)

def gc_content(sequence):
    """
    Calculates the GC content as a percentage in the RNA sequence.
    
    Args:
    sequence (str): The RNA sequence to analyze.
    
    Returns:
    float: The GC content percentage.
    """
    gc_count = sequence.count('G') + sequence.count('C')
    return (gc_count / len(sequence)) * 100 if sequence else 0

# Test the functions with the provided examples
test_sequence = "AUGCAUGCAUGC"
test_motif = "AUG"
print(f"is_valid_RNA: {is_valid_rna(test_sequence)}")
print(f"Nucleotide count: {nucleotide_count(test_sequence)}")
print(f"Motif '{test_motif}' found at positions: {find_motifs(test_sequence, test_motif)}")
print(f"Complementary sequence: {complementary_sequence(test_sequence)}")
print(f"GC content: {gc_content(test_sequence):.1f} %")

# Part II

def is_valid_rna_advanced(sequence):
    """
    Checks if a given string is a valid RNA sequence, including ambiguity codes.
    
    Args:
    sequence (str): The RNA sequence to validate.
    
    Returns:
    bool: True if the sequence is valid, False otherwise.
    """
    valid_chars = 'ACGURYKMSWBDHVN'
    return all(char in valid_chars for char in sequence)

def find_motifs_with_ambiguity(sequence, motif):
    """
    Identifies occurrences of a motif with ambiguity codes within the RNA sequence.
    
    Args:
    sequence (str): The RNA sequence to search.
    motif (str): The motif (with ambiguity codes) to find.
    
    Returns:
    list: A list of positions where the motif is found.
    """
    ambiguity_codes = {
        'R': '[AG]', 'Y': '[CTU]', 'S': '[CG]', 'W': '[ATU]',
        'K': '[GTU]', 'M': '[AC]', 'B': '[CGTU]', 'D': '[AGTU]',
        'H': '[ACTU]', 'V': '[ACG]', 'N': '[ACGTU]'
    }
    regex_motif = ''.join(ambiguity_codes.get(char, char) for char in motif)
    return [match.start() for match in re.finditer(regex_motif, sequence)]

def fragment_and_analyze(sequence, fragment_length):
    """
    Fragments the RNA sequence and performs a detailed analysis on each fragment.
    
    Args:
    sequence (str): The RNA sequence to fragment.
    fragment_length (int): The length of each fragment.
    
    Returns:
    list: A list of dictionaries containing analysis results for each fragment.
    """
    fragments = [sequence[i:i+fragment_length] for i in range(0, len(sequence), fragment_length)]
    analysis_results = []
    for fragment in fragments:
        result = {
            'fragment': fragment,
            'is_valid_rna': is_valid_rna_advanced(fragment),
            'gc_content': gc_content(fragment) if is_valid_rna_advanced(fragment) else 'N/A',
            'complementary_sequence': complementary_sequence(fragment) if is_valid_rna_advanced(fragment) else 'N/A'
        }
        analysis_results.append(result)
    return analysis_results

# Test the functions with the provided examples
test_sequence_advanced = "AUGCRYSWKMBDHVN"
test_invalid_sequence = "AUGTXZGCAUGC"
test_motif_ambiguous = "RY"
print(f"Advanced is_valid_RNA: {is_valid_rna_advanced(test_sequence_advanced)}")
print(f"Advanced is_valid_RNA (invalid): {is_valid_rna_advanced(test_invalid_sequence)}")
print(f"Motif with ambiguity '{test_motif_ambiguous}' found at positions: {find_motifs_with_ambiguity(test_sequence_advanced, test_motif_ambiguous)}")

fragment_test_sequence = "AUGCRYSNAUGCRYXNAUGCRYSN"
fragment_length = 6
print("Fragment Analysis:")
for result in fragment_and_analyze(fragment_test_sequence, fragment_length):
    print(result)