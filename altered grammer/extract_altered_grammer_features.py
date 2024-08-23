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
# from nltk import Tree

os.environ['STANFORD_PARSER'] = '/Users/rohithperumandla/Desktop/Research/jar'
os.environ['STANFORD_MODELS'] = '/Users/rohithperumandla/Desktop/Research/jar'
parser = stanford.StanfordParser(model_path="/Users/rohithperumandla/Desktop/Research/jar/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

def count_characters(text):
    # Simply return the length of the text
    return len(text)
    


def extract_function_words(tree):
    # Function words POS tags according to Penn Treebank
    function_word_tags = {
        'CC', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB'
    }
    
    function_words = []
    # print('tree',tree)
    # print("tree.subtrees()",tree.subtrees())
    for subtree in tree.subtrees():
        # print('subtree',subtree)
        # print("subtree.label()",subtree.label())
        if subtree.label() in function_word_tags:
            function_words.append(subtree.leaves()[0])
    
    return len(function_words)

def pos_tags_features(sentences_list):
    total_coordinated_sentence = 0
    total_subordinated_sentence = 0
    total_reduced_sentence = 0
    total_production_rules  = 0
    total_function_words = 0
    
    for data in sentences_list:
        sentences = parser.raw_parse_sents([data])
        for line in sentences:
            for sentence in line:
                data = str(sentence)
                tree = Tree.fromstring(data)
                production_rules = extract_production_rules(tree)
                function_words = extract_function_words(tree)
                total_production_rules  +=production_rules
                total_function_words +=function_words
                
                def extract_tags(tree):
                    tag_data = {}
                    def traverse(subtree):
                        tag = subtree.label()
                        if tag not in tag_data:
                            tag_data[tag] = {'count': 0, 'words': []}
                        tag_data[tag]['count'] += 1
                        tag_data[tag]['words'].extend(subtree.leaves())
                        
                        for child in subtree:
                            if isinstance(child, Tree):
                                traverse(child)
                    traverse(tree)
                    return tag_data
                
                # Extract tag data
                tag_data = extract_tags(tree)
                try:
                    cs_count = tag_data['CC']['count']
                except:
                    cs_count = 0                    

                try:        
                    ss_count = tag_data['S']['count']
                except:
                    ss_count = 0                    
        
                try:
                    vbg = tag_data['VBG']['count']
                except:                    
                    vbg = 0
                    
                try:
                    vbn = tag_data['VBN']['count']        
                except:
                    vbn = 0
        
                try:
                    vbz = tag_data['VBZ']['count'] 
                except:
                    vbz = 0
                    
                rs_count = vbn  +vbg
        
                total_coordinated_sentence += cs_count
                total_subordinated_sentence += ss_count
                total_reduced_sentence +=rs_count
    
    return total_coordinated_sentence, total_subordinated_sentence, total_reduced_sentence, production_rules, function_words
#------------------------------------------------------------------#------------------------------------------------------------------

# Counts Predicates

# Function to count predicates
def count_predicates(doc):
    predicate_count = 0
    
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos == 'VERB':
                for dep in sentence.dependencies:
                    if (dep[0].id == word.id or dep[2].id == word.id) and dep[1] in {'nsubj', 'obj', 'iobj', 'ccomp', 'xcomp'}:
                        # print('Matching dependency:', dep)
                        predicate_count += 1
                        break  # Count each predicate only once
    return predicate_count


#------------------------------------------------------------------#------------------------------------------------------------------


def extract_production_rules(tree):
    # Extract the production rules from the tree
    productions = tree.productions()
    
    # Convert to a set to ensure uniqueness
    unique_productions = set(productions)
    
    # Define symbols to remove
    symbols_to_remove = { ',', '.'}
    
    # Filter out productions containing the specified symbols
    filtered_productions = {rule for rule in unique_productions if not any(symbol in str(rule) for symbol in symbols_to_remove)}
    
    return len(filtered_productions)

#------------------------------------------------------------------#------------------------------------------------------------------


def tokenize(text):
    """Tokenize the text into words."""
    return re.findall(r'\b\w+\b', text.lower())

def count_unique_words(text):
    """Count the total number of unique words in the text,
       excluding immediately repeated words."""
    words = tokenize(text)
    unique_words = set()

    
    # Iterate through the words and count unique words
    for i, word in enumerate(words):
        unique_words.add(word)

    return len(unique_words), len(words)


#------------------------------------------------------------------#------------------------------------------------------------------


def remove_commas_and_periods(text):
    # Remove commas and periods using regular expressions
    text = re.sub(r'[,.]', '', text)
    return text


def count_immediate_repetitions(text):
    text = remove_commas_and_periods(text)
    # Split the text into words
    words = text.split()


    # Initialize counter for immediate repetitions
    repetitions = 0

    # Iterate through the words, comparing each word with the next word
    for i in range(len(words)-1):

        if words[i] == words[i + 1]:
            
            repetitions += 1

    return repetitions
