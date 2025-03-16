import pandas as pd
import os
import re
import textstat
import stanza
import numpy as np

# Download and load Stanza model (only needed first time)
# stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse')

# Load words from CSV files
def load_word_list(filename):
    """Loads a list of words from a CSV file."""
    file_path = os.path.join(filename)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)  # Assumes a single column named 'word'
        return set(df['word'].dropna().str.lower())  # Convert to lowercase and remove NaNs
    return set()  # Return an empty set if file doesn't exist

# Load words from CSV files
USER_DIRECT_VERBS = load_word_list("final_unique_behaviours.csv")
INTERFACE_ELEMENT_WORDS = load_word_list("interface_element_words.csv") or load_word_list("interactive_elements.csv")
SYSTEM_DEFECT_WORDS = load_word_list("_negative_words.csv")

# Define fallback word lists in case CSV files are not found
ENVIRONMENT_WORDS = {"android", "ios", "chrome", "firefox", "windows", "macos"}
SCREENSHOT_WORDS = {"screenshot", "attached image", "see attachment"}
SCREENSHOT_GUIDELINE_WORDS = {"please see", "as shown in the image", "attachment"}
if not INTERFACE_ELEMENT_WORDS:
    INTERFACE_ELEMENT_WORDS = {"adapter", "button", "page", "menu", "dialog", "tab", "screen"}
if not SYSTEM_DEFECT_WORDS:
    SYSTEM_DEFECT_WORDS = {"crash", "down", "flashback", "overlap", "too big", "too small", "freeze"}
if not USER_DIRECT_VERBS:
    USER_DIRECT_VERBS = {"login", "register", "logout"}

# --------------------------------------------------------------------
# HELPER FUNCTIONS FOR RULE CHECKS
# --------------------------------------------------------------------

# ============= MORPHOLOGICAL INDICATORS =============
def check_RM1_size(report_text, min_tokens=50, max_tokens=300):
    """
    Check if the text length (in tokens) is within a preset range.
    For example, we say 50-300 tokens is acceptable.
    """
    tokens = report_text.split()
    if min_tokens <= len(tokens) <= max_tokens:
        return True, 1  # (passed, 1 point)
    return False, 0

def check_RM2_readability(report_text, min_score=30, max_score=100):
    """
    Computes the Flesch Reading Ease Score for the given text.
    Returns True if the score is within the specified range, along with the score itself.
    """
    if not report_text.strip():
        return False, 0
    
    flesch_score = textstat.flesch_reading_ease(report_text)
    return min_score <= flesch_score <= max_score, 1 if min_score <= flesch_score <= max_score else 0

def check_RM3_punctuation(report_text):
    """
    Check if punctuation is properly used.
    A naive approach might check if each sentence ends with . ? or !
    """
    # Simple check: all sentences in text end with punctuation
    sentences = re.split(r'[.!?]', report_text.strip())
    # if the last split is empty, ignore it
    sentences = [s.strip() for s in sentences if s.strip()]
    for s in sentences:
        # We append a '.' to ensure we have something to match
        if not re.match(r'.+[.!?]$', s + "."):
            return False, 0
    return True, 1

def check_RM4_avg_sentence_length(doc, min_len=5, max_len=40):
    """
    Check the average length of a sentence is within a preset range.
    We use spaCy's doc.sents to measure sentence lengths in tokens.
    """
    sents = list(doc.sentences)
    if not sents:
        return False, 0
    lengths = [len(sent.words) for sent in sents]
    avg_len = sum(lengths)/len(lengths)
    if min_len <= avg_len <= max_len:
        return True, 1
    return False, 0

# ============= RELATIONAL INDICATORS =============
def check_RR1_itemization(report_text):
    """
    Check if there's a step-by-step description. 
    For simplicity, check if text has bullet points or numbering (like 1) or (2).
    """
    # naive approach: look for bullet or numbering patterns
    if re.search(r'^(\d+\)|-|\*)', report_text, flags=re.MULTILINE):
        return True, 1
    return False, 0

def check_RR2_itemization_symbol(report_text):
    """
    Check if the itemization symbol follows a specification.
    For example, maybe only '-' or '*' or numeric is allowed.
    """
    # Suppose we say it passes if we find lines starting with '-' or '*'
    if re.search(r'^(-|\*)', report_text, flags=re.MULTILINE):
        return True, 1
    return False, 0

def check_RR3_environment(report_text):
    """
    Check if environmental information is present.
    We'll pass if we find any environment-related keywords in the text.
    """
    text_lower = report_text.lower()
    for word in ENVIRONMENT_WORDS:
        if word in text_lower:
            return True, 2  # environment is fairly important => 2 points
    return False, 0

def check_RR4_screenshot(report_text):
    """
    Check if there's at least one screenshot mention.
    """
    text_lower = report_text.lower()
    for w in SCREENSHOT_WORDS:
        if w in text_lower:
            return True, 1
    return False, 0

def check_RR5_screenshot_guideline(report_text):
    """
    Check if guideline words for screenshot exist (like "Please see attached screenshot" etc.).
    """
    text_lower = report_text.lower()
    for w in SCREENSHOT_GUIDELINE_WORDS:
        if w in text_lower:
            return True, 1
    return False, 0

# ============= RELATIONAL ANALYSIS FUNCTIONS =============
def get_phrase_with_modifiers(sentence, word_id):
    """Extract a phrase including the word and its modifiers."""
    word = sentence.words[word_id - 1]
    modifiers = [w.text for w in sentence.words if w.head == word_id]
    phrase = word.text + ' ' + ' '.join(modifiers)
    return phrase

def has_location_preposition(sentence, word_id):
    """Check if a word has a location preposition attached to it."""
    for w in sentence.words:
        if w.head == word_id and w.upos == 'ADP' and w.text.lower() in ['in', 'on', 'at']:
            return True
    return False

def has_time_preposition(sentence, word_id):
    """Check if a word has a time preposition attached to it."""
    for w in sentence.words:
        if w.head == word_id and w.upos == 'ADP' and w.text.lower() in ['during', 'after', 'before']:
            return True
    return False

def check_RA1_interface_element(text):
    """
    Check for a clear and unambiguous interface element description using Stanza.
    Exactly follows the specified relation types: ATT, ADV, CMP, COO, LAD, RAD.
    Returns True with 2 points if enough relations are found.
    """
    # Process with Stanza for detailed dependency parsing
    doc = nlp(text)
    
    # Check if any interface element word exists in the text
    text_lower = text.lower()
    if not any(w in text_lower for w in INTERFACE_ELEMENT_WORDS):
        return False, 0
    
    # Mapping from Universal Dependencies to the specific relation types
    # based on the requirements
    relation_mapping = {
        # Modifying relationships
        'ATT': ['amod', 'nmod', 'nummod', 'det', 'compound', 'case', 'mark'],  # Attributes
        'ADV': ['advmod', 'advcl', 'obl', 'neg'],  # Adverbials
        'CMP': ['obj', 'iobj', 'xcomp', 'ccomp', 'acl'],  # Complements
        
        # Complementary relationships
        'COO': ['conj', 'cc'],  # Coordinate
        'LAD': ['acl:relcl', 'acl'],  # Left adjunct
        'RAD': ['appos', 'parataxis']  # Right adjunct
    }
    
    # Flatten the relation mapping for easier lookup
    relation_lookup = {}
    for rel_type, ud_rels in relation_mapping.items():
        for ud_rel in ud_rels:
            relation_lookup[ud_rel] = rel_type
    
    # Track scores for interface elements
    interface_elements_found = False
    max_relation_score = 0
    
    for sentence in doc.sentences:
        # Find interface elements in the sentence
        for word in sentence.words:
            if word.lemma.lower() in INTERFACE_ELEMENT_WORDS:
                interface_elements_found = True
                
                # Count the different types of relations this interface element has
                relations = {
                    'ATT': 0, 'ADV': 0, 'CMP': 0,  # Modifying
                    'COO': 0, 'LAD': 0, 'RAD': 0   # Complementary
                }
                
                # Check incoming relations (dependencies where this word is the head)
                for other_word in sentence.words:
                    if other_word.head == word.id:  # This word is the head of the dependency
                        rel_type = relation_lookup.get(other_word.deprel, None)
                        if rel_type:
                            relations[rel_type] += 1
                
                # Check outgoing relations (where this word is dependent)
                if word.head != 0:  # Not the root
                    head_word = sentence.words[word.head - 1]
                    rel_type = relation_lookup.get(word.deprel, None)
                    if rel_type:
                        relations[rel_type] += 1
                
                # Calculate the relation score - different combinations as per requirements
                modifying_count = relations['ATT'] + relations['ADV'] + relations['CMP']
                complementary_count = relations['COO'] + relations['LAD'] + relations['RAD']
                total_relation_count = modifying_count + complementary_count
                
                # Calculate score based on relation counts
                relation_score = 0
                if total_relation_count >= 4:
                    relation_score = 2  # Full score for detailed description
                elif total_relation_count >= 2:
                    relation_score = 1  # Partial score for some description
                
                max_relation_score = max(max_relation_score, relation_score)
    
    return interface_elements_found and max_relation_score > 0, max_relation_score

def check_RA2_user_behavior(text):
    """
    Check for presence of user interaction behavior descriptions using Stanza.
    Categories checked:
    1. User-interface interaction (action with interface element)
    2. Time-space change descriptions
    3. Direct system interaction verbs
    
    Returns True with appropriate points if user behaviors are found.
    """
    # Process with Stanza for detailed parsing
    doc = nlp(text)
    text_lower = text.lower()
    
    # Category 3: Check for direct system interaction verbs first (simplest check)
    if any(verb in text_lower for verb in USER_DIRECT_VERBS):
        return True, 2  # Full score for direct system interaction
    
    # Check other categories which require deeper analysis
    for sentence in doc.sentences:
        # Find predicates (verbs) in the sentence
        predicates = [word for word in sentence.words if word.upos == 'VERB']
        
        # Category 1: User-interface interaction
        for verb in predicates:
            # Check if the verb has direct objects (obj dependency)
            obj_deps = [w for w in sentence.words if w.head == verb.id and w.deprel in ['obj', 'iobj']]
            
            for obj in obj_deps:
                # Check if the object or its modifiers contain interface element words
                obj_phrase = get_phrase_with_modifiers(sentence, obj.id)
                if any(element in obj_phrase.lower() for element in INTERFACE_ELEMENT_WORDS):
                    return True, 2  # Full score for user-interface interaction
        
        # Category 2: Time-space change descriptions
        has_loc = False
        has_tmp = False
        
        # Check for location indicators
        loc_deps = [w for w in sentence.words if w.deprel in ['obl', 'advmod'] and 
                    any(loc_word in w.text.lower() or 
                        has_location_preposition(sentence, w.id) 
                        for loc_word in ['in', 'on', 'at', 'page', 'screen', 'window'])]
        
        if loc_deps:
            has_loc = True
        
        # Check for time indicators
        tmp_deps = [w for w in sentence.words if w.deprel in ['obl', 'advmod'] and 
                    any(tmp_word in w.text.lower() or 
                        has_time_preposition(sentence, w.id)
                        for tmp_word in ['during', 'when', 'while', 'after', 'before'])]
        
        if tmp_deps:
            has_tmp = True
        
        # If both time and space indicators are present
        if has_loc and has_tmp:
            return True, 2  # Full score for time-space description
        elif has_loc or has_tmp:
            return True, 1  # Partial score for either time or space description
    
    return False, 0  # No user behavior description found

def check_RA3_system_defect(text):
    """
    Check for presence of system defect descriptions using Stanza.
    Categories checked:
    1. "Negation + action" describing errors
    2. Explicit error words or layout defect terms
    
    Returns True with appropriate points if system defects are found.
    """
    # Process with Stanza for detailed parsing
    doc = nlp(text)
    text_lower = text.lower()
    
    # Category 2: Check for explicit system defect words (simplest check)
    if any(defect in text_lower for defect in SYSTEM_DEFECT_WORDS):
        return True, 2  # Full score for explicit system defect
    
    # Category 1: "Negation + action" describing errors
    for sentence in doc.sentences:
        # Find predicates (actions) in the sentence
        predicates = [word for word in sentence.words if word.upos == 'VERB']
        
        for verb in predicates:
            # Check if the action has a negation marker
            has_negation = any(w.deprel == 'advmod' and w.text.lower() in ['not', "n't", 'never', 'no'] 
                              for w in sentence.words if w.head == verb.id)
            
            if has_negation:
                # Check if this is a user action or system action
                verb_phrase = get_phrase_with_modifiers(sentence, verb.id)
                
                # Look for indicators that this is a system action (like "load", "display", "show")
                system_actions = ["load", "display", "show", "appear", "refresh", "update", "work", "function"]
                is_system_action = any(action in verb.lemma.lower() for action in system_actions)
                
                # Check if any object of the verb is an interface element
                has_interface_object = any(w.deprel == 'obj' and 
                                         any(elem in w.lemma.lower() for elem in INTERFACE_ELEMENT_WORDS)
                                         for w in sentence.words if w.head == verb.id)
                
                if is_system_action or has_interface_object:
                    return True, 2  # Full score for negated system action
                else:
                    return True, 1  # Partial score for negated action (might be user action)
    
    return False, 0  # No system defect description found

def check_RA4_defect_description_quality(text):
    """
    Check for clear and unambiguous description of system defects.
    Similar to RA1, using modifying and complementary relations.
    
    Returns True with appropriate points based on relation quality.
    """
    # First check if there's a system defect present at all
    has_defect, _ = check_RA3_system_defect(text)
    if not has_defect:
        return False, 0  # No defect to describe
    
    # Process with Stanza for detailed dependency parsing
    doc = nlp(text)
    
    # Mapping from Universal Dependencies to the specific relation types
    # Same as used in RA1
    relation_mapping = {
        # Modifying relationships
        'ATT': ['amod', 'nmod', 'nummod', 'det', 'compound', 'case', 'mark'],  # Attributes
        'ADV': ['advmod', 'advcl', 'obl', 'neg'],  # Adverbials
        'CMP': ['obj', 'iobj', 'xcomp', 'ccomp', 'acl'],  # Complements
        
        # Complementary relationships
        'COO': ['conj', 'cc'],  # Coordinate
        'LAD': ['acl:relcl', 'acl'],  # Left adjunct
        'RAD': ['appos', 'parataxis']  # Right adjunct
    }
    
    # Flatten the relation mapping for easier lookup
    relation_lookup = {}
    for rel_type, ud_rels in relation_mapping.items():
        for ud_rel in ud_rels:
            relation_lookup[ud_rel] = rel_type
    
    # Find defect terms or negated actions
    max_relation_score = 0
    
    for sentence in doc.sentences:
        # Find system defect words
        defect_words = []
        
        # Add explicit defect terms
        for word in sentence.words:
            if word.lemma.lower() in SYSTEM_DEFECT_WORDS:
                defect_words.append(word)
        
        # Add negated verbs that might be system actions
        for word in sentence.words:
            if word.upos == 'VERB':
                has_negation = any(w.deprel == 'advmod' and w.text.lower() in ['not', "n't", 'never', 'no'] 
                                  for w in sentence.words if w.head == word.id)
                if has_negation:
                    defect_words.append(word)
        
        # No defect words found in this sentence
        if not defect_words:
            continue
        
        # For each defect word, check its relations
        for defect_word in defect_words:
            # Count the different types of relations this defect term has
            relations = {
                'ATT': 0, 'ADV': 0, 'CMP': 0,  # Modifying
                'COO': 0, 'LAD': 0, 'RAD': 0   # Complementary
            }
            
            # Check incoming relations (dependencies where this word is the head)
            for other_word in sentence.words:
                if other_word.head == defect_word.id:
                    rel_type = relation_lookup.get(other_word.deprel, None)
                    if rel_type:
                        relations[rel_type] += 1
            
            # Check outgoing relations (where this word is dependent)
            if defect_word.head != 0:  # Not the root
                rel_type = relation_lookup.get(defect_word.deprel, None)
                if rel_type:
                    relations[rel_type] += 1
            
            # Calculate the relation score - different combinations as per requirements
            modifying_count = relations['ATT'] + relations['ADV'] + relations['CMP']
            complementary_count = relations['COO'] + relations['LAD'] + relations['RAD']
            total_relation_count = modifying_count + complementary_count
            
            # Calculate score based on relation counts
            relation_score = 0
            if total_relation_count >= 4:
                relation_score = 2  # Full score for detailed description
            elif total_relation_count >= 2:
                relation_score = 1  # Partial score for some description
            
            max_relation_score = max(max_relation_score, relation_score)
    
    return max_relation_score > 0, max_relation_score

def evaluate_bug_report(text):
    """Evaluate the overall quality of a bug report."""
    # Process with Stanza
    doc = nlp(text)
    
    # Morphological checks
    results = {
        "RM1_size": check_RM1_size(text),
        "RM2_readability": check_RM2_readability(text),
        "RM3_punctuation": check_RM3_punctuation(text),
        "RM4_sentence_length": check_RM4_avg_sentence_length(doc),
        
        # Relational checks
        "RR1_itemization": check_RR1_itemization(text),
        "RR2_itemization_symbol": check_RR2_itemization_symbol(text),
        "RR3_environment": check_RR3_environment(text),
        "RR4_screenshot": check_RR4_screenshot(text),
        "RR5_screenshot_guideline": check_RR5_screenshot_guideline(text),
        
        # Relational analysis checks
        "RA1_interface_element": check_RA1_interface_element(text),
        "RA2_user_behavior": check_RA2_user_behavior(text),
        "RA3_system_defect": check_RA3_system_defect(text),
        "RA4_defect_description": check_RA4_defect_description_quality(text)
    }
    
    # Calculate overall score
    total_score = sum(score for _, score in results.values())
    
    # Calculate maximum possible score
    max_possible = 16  # Sum of all maximum scores for each check
    
    return {
        "detail_scores": results,
        "total_score": total_score,
        "max_possible": max_possible
    }

def process_excel_file(excel_path, output_path=None,output_prefix="bug_report_scores"):
    """
    Process an Excel file with bug reports and evaluate each one.
    
    Args:
        excel_path: Path to the Excel file containing bug reports
        output_path: Optional path to save the results
        
    Returns:
        DataFrame with original data and evaluation scores
    """
    # Load the Excel file
    try:
        df = pd.read_excel(excel_path)
        print(f"Number of rows read from Excel: {len(df)}") 
        # df = df[:5]
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None
    
    # Assume the bug report text is in a column named 'bug_report' or similar
    # Adjust this column name as needed
    text_column = None
    possible_columns = ['4o_gpt Output']
    
    for col in possible_columns:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        if len(df.columns) > 0:
            # Use the first column if none of the expected columns are found
            text_column = df.columns[0]
        else:
            print("No suitable text column found in the Excel file")
            return None
    
    # Evaluate each bug report
    results = []
    file_count = 1
    for idx, row in df.iterrows():
        print("doing index",idx)
        report_text = str(row[text_column])
        evaluation = evaluate_bug_report(report_text)
        
        # Create a result dictionary with the original row data
        result = dict(row)
        
        # Add evaluation scores
        result['total_score'] = evaluation['total_score']
        result['max_possible'] = evaluation['max_possible']
        result['score_percentage'] = (evaluation['total_score'] / evaluation['max_possible']) * 100
        
        # Add detailed scores
        for rule, (passed, score) in evaluation['detail_scores'].items():
            result[f'{rule}_passed'] = passed
            result[f'{rule}_score'] = score

        if (idx + 1) % 1000 == 0 or idx == len(df) - 1:  # Save every 1000 rows
            results_df = pd.DataFrame(results)
            output_file = f"./Original_dataset/{output_prefix}_score_here_all_12k_{file_count}.xlsx"
            results_df.to_excel(output_file, index=False)
            print(f"Saved {len(results)} rows to {output_file}")
            results.clear()  # Clear the list for the next batch
            file_count += 1
        
        results.append(result)
    
    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)
    
    # Save to output file if specified
    if output_path:
        results_df.to_excel(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    return results_df

# Main function to run when script is executed directly
if __name__ == "__main__":
    import sys
    
    # if len(sys.argv) < 2:
    #     print("Usage: python script.py input.xlsx [output.xlsx]")
    #     sys.exit(1)
    
    input_file = "Final_12k_good_reports_Ollama.xlsx"
    output_file = "TEst_FInal_all_12k_bug_report_scores12000.xlsx"
    
    result = process_excel_file(input_file, output_file)
        
    if result is not None:
        print(f"Processed {len(result)} bug reports.")
        print(f"Average score: {result['score_percentage'].mean():.2f}%")