import re

def extract_fillers_and_rate(input_data, duration_seconds):
    # Ensure the input is processed as a list of strings
    if isinstance(input_data, str):
        sentences = [input_data]  # Convert single string to list
    elif isinstance(input_data, list):
        sentences = input_data
    else:
        raise ValueError("Input must be a string or a list of strings.")
    
    # Regular expression pattern to match filler words composed of 'u', 'm', 'h', and 'a'
    pattern = r'\b[umha]+\b'
    
    # Collect all filler words from all sentences
    all_fillers = []
    for sentence in sentences:
        filler_words = re.findall(pattern, sentence, re.IGNORECASE)
        all_fillers.extend(filler_words)
    
    # Convert duration from seconds to minutes
    duration_minutes = duration_seconds / 60
    
    # Calculate the rate of filler words per minute
    if duration_minutes > 0:
        fillers_per_minute = len(all_fillers) / duration_minutes
    else:
        fillers_per_minute = 0 
    
    return all_fillers, fillers_per_minute