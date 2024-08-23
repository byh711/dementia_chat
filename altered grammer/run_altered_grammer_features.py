from nltk.parse import stanford
import os
from nltk.tree import Tree
import re
from collections import Counter
import nltk
import stanza
import pandas as pd
import ast
import re
from datetime import datetime

from extract_altered_grammer_features import pos_tags_features,count_predicates, count_unique_words, count_immediate_repetitions, count_characters

# Initialize the pipeline for predicates
nlp = stanza.Pipeline('en')

df = pd.read_excel('dementia_data.xlsx')
all_sentences = df['sentence'].apply(ast.literal_eval)

extracted_features = {"File Name":[],"coordinated_sentence":[],"subordinated_sentence":[],"reduced_sentence":[],"predicates":[], "production_rules":[],"function_words":[],"unique_words":[],"total_words":[],"character_length":[],"immediate_word_repetitions":[]}
# Iterate over DataFrame rows
for index, row in df.iterrows():
    file_name = row['file']
    list_sentences = ast.literal_eval(row['sentence'])
    total_coordinated_sentence,total_subordinated_sentence,total_reduced_sentence, production_rules,function_words =  pos_tags_features(list_sentences)
    print(total_coordinated_sentence,total_subordinated_sentence,total_reduced_sentence)

    extracted_features["File Name"].append(file_name)
    extracted_features["coordinated_sentence"].append(total_coordinated_sentence)
    extracted_features["subordinated_sentence"].append(total_subordinated_sentence)
    extracted_features["reduced_sentence"].append(total_reduced_sentence)
    extracted_features["production_rules"].append(production_rules)
    extracted_features["function_words"].append(function_words)
    
    
#------------------------------------------------------------------#------------------------------------------------------------------
# Predicates calling
    
    overall_total_predicates = 0
    total_immediate_word_repetitions=0
    total_text_character_length =0
    total_unique_words = 0
    total_words = 0
    
    for data in list_sentences:
    # Process the text
        doc = nlp(data)
        # Count predicates in the processed document
        predicate_count = count_predicates(doc)

        # Count unique words
        unique_words, words = count_unique_words(data)  # # Unique words and total words
        immediate_word_repetitions = count_immediate_repetitions(data)
        text_character_length = count_characters(data)

        total_immediate_word_repetitions+=immediate_word_repetitions
        total_text_character_length +=text_character_length
        total_unique_words+= unique_words
        total_words+=words
        overall_total_predicates+=predicate_count

    
    extracted_features["predicates"].append(overall_total_predicates)
    extracted_features["unique_words"].append(total_unique_words)
    extracted_features["total_words"].append(total_words)
    extracted_features["character_length"].append(total_text_character_length)
    extracted_features["immediate_word_repetitions"].append(total_immediate_word_repetitions)
    
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create the filename with the timestamp
filename = f"dementia_extracted_feature_{timestamp}.xlsx"
df = pd.DataFrame(extracted_features)
# Save the DataFrame to an Excel file with the timestamped filename
df.to_excel(filename, index=False)
print(f"File saved as {filename}")



