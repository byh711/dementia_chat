import os
import re
from datetime import datetime

def extract_timestamp(filename):
    # Regular expression to extract date and time from the filename
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        # Convert the extracted timestamp into a datetime object
        return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
    return None


def find_most_recent_file(directory, prefix='logistic_regression_model', extension='.pkl'):
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
    
    if not files:
        return None  # No files found matching the criteria
    
    # Extract timestamps and find the most recent file
    most_recent_file = max(files, key=lambda f: extract_timestamp(f))
    
    return most_recent_file




