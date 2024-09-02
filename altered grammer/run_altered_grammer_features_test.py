import os
import nltk
import stanza
import pandas as pd
import ast
import re
from datetime import datetime
import numpy as np
from extract_altered_grammer_features import pos_tags_features,count_predicates, count_unique_words, count_immediate_repetitions, count_characters
from ml_model.altered_grammer_score import calculate_probability
nlp = stanza.Pipeline('en')

"""get_altered_grammer_features function will take a list of sentences and the total speech duration
    of the sentences and returns the altered grammer score/probility. In the below function it calls calculate_probability
    where  the weights of the trained ML model and these features are used to calculate the score."""

def get_altered_grammer_features(list_sentences,speech_duration_seconds):

    extracted_features = {"File Name":[],"coordinated_sentence":[],"subordinated_sentence":[],"reduced_sentence":[],"predicates":[], "production_rules":[],"function_words":[],"unique_words":[],"total_words":[],"character_length":[],"immediate_word_repetitions":[]}
    
    # for index, row in df.iterrows():
    #     file_name = row['file']
    #     list_sentences = ast.literal_eval(row['sentence'])
    total_coordinated_sentence,total_subordinated_sentence,total_reduced_sentence, production_rules,function_words =  pos_tags_features(list_sentences)
    print(total_coordinated_sentence,total_subordinated_sentence,total_reduced_sentence)

    # extracted_features["File Name"].append(file_name)
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
    total_time_minutes = speech_duration_seconds / 60  # Convert to minutes
    df = pd.DataFrame(extracted_features)
    first_row = df.iloc[0]
    extracted_features = first_row.to_numpy()

    extracted_features_per_minute = extracted_features / total_time_minutes

    altered_grammer_score = calculate_probability(extracted_features_per_minute)

    return altered_grammer_score
    # # Create the filename with the timestamp
    # filename = f"dementia_extracted_feature_{timestamp}.xlsx"
    
    # # Save the DataFrame to an Excel file with the timestamped filename
    # df.to_excel(filename, index=False)
    # print(f"File saved as {filename}")



