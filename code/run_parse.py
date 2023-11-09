import argparse
import regex
import json
import unicodedata
import csv
import pandas as pd
import spacy
from booknlp.booknlp import BookNLP

# I/O Arguments

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--example_file', type=str, required=True)
arg_parser.add_argument('--output_file', type=str, required=True)
arg_parser.add_argument('--working_dir', type=str, required=True)

args = arg_parser.parse_args()

# BookNLP is going to create temporary files

error_fn = os.path.join(args.working_dir,'error.txt')
temp_fn = os.path.join(args.working_dir,'temp-file.txt')
temp_dir = os.path.join(args.working_dir,'temp-dir/')

# Initialize Models

model_params={"pipeline":"entity,supersense", "model":"big"}
booknlp=BookNLP("en", model_params)

nlp = spacy.load('en_core_web_sm')

# Controlled Vocabulary of NLP Features

pos_tags = ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN','NUM',
            'PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X']

sst_tags = ['noun.act','noun.animal','noun.artifact','noun.attribute',
            'noun.body','noun.cognition','noun.communication','noun.event',
            'noun.feeling','noun.food','noun.group','noun.location',
            'noun.motive','noun.object','noun.person','noun.phenomenon',
            'noun.plant','noun.possession','noun.process','noun.quantity',
            'noun.relation','noun.shape','noun.state','noun.substance',
            'noun.time','noun.Tops',
            'verb.body','verb.change','verb.cognition','verb.communication',
            'verb.competition','verb.consumption','verb.contact',
            'verb.creation','verb.emotion','verb.motion','verb.perception',
            'verb.possession','verb.social','verb.stative','verb.weather']

mor_tags = ['Aspect=Perf', 'Aspect=Prog', 'Case=Acc', 'Case=Nom',
            'ConjType=Cmp', 'Definite=Def', 'Definite=Ind', 'Degree=Cmp',
            'Degree=Pos', 'Degree=Sup', 'Foreign=Yes', 'Gender=Fem',
            'Gender=Masc', 'Gender=Neut', 'Hyph=Yes', 'Mood=Ind', 'NumType=Card',
            'NumType=Mult', 'NumType=Ord', 'Number=Plur', 'Number=Sing', 'Person=1',
            'Person=2', 'Person=3', 'Polarity=Neg', 'Poss=Yes', 'PronType=Art',
            'PronType=Dem', 'PronType=Ind', 'PronType=Prs', 'PronType=Rel',
            'PunctSide=Fin', 'PunctSide=Ini', 'PunctType=Brck', 'PunctType=Comm',
            'PunctType=Dash', 'PunctType=Peri', 'PunctType=Quot', 'Reflex=Yes',
            'Tense=Past', 'Tense=Pres', 'VerbForm=Fin', 'VerbForm=Ger', 'VerbForm=Inf',
            'VerbForm=Part', 'VerbType=Mod', '_']

ft_list = pos_tags + sst_tags + mor_tags

# Unicode Management

corner_case_list = []
#corner_case_list = ['In multi-hop networks','The letters Ε (epsilon) and Θ (theta)','At the welcoming feast','Antipater II of Macedon']

def remove_accents(str_in):

    # Initialize variables
    map_out_to_in = []
    map_d = 0
    str_out = ''

    for c in str_in:

        c_norm = unicodedata.normalize('NFD',c)

        # BERT keeps vowels in Korean, Tamil, Kannada, Bengali
        if (ord(c) >= 44032 and ord(c) <= 55203) or (ord(c) >= 2946 and ord(c) <= 3064 ) or (ord(c) >= 3200 and ord(c) <= 3315 ) or (ord(c) >= 2432 and ord(c) <= 2558 ):
            new_c = c_norm
        # Remove Formatting Characters
        elif unicodedata.category(c) in set(['Co','Cf','Cc','Cn']):
            new_c = " "
        else:
            # Remove Diacritics
            new_c = list(c_norm)[0]
        
        str_out += new_c

        # Create Mapping from un-accented output to initial input
        map_out_to_in.append(map_d)
        for _ in new_c[1:]:
            map_d -= 1
            map_out_to_in.append(map_d)
    
    # Update Mapping for Remaining Diacritics
    d_index_list = [dex for d in regex.finditer(r"\p{Mn}", str_out) for dex in range(d.start(0),d.end(0))]

    total_popped = 0
    for d_index in d_index_list:
        map_out_to_in.pop(d_index - total_popped)
        for new_dex in range(d_index - total_popped,len(map_out_to_in)):
            map_out_to_in[new_dex] += 1
        total_popped += 1
    
    # Range is "exclusive" so needs additional value at end
    map_out_to_in.append( map_out_to_in[-1] )

    # Remove Remaining Diacritics
    str_out = regex.sub(r"\p{Mn}","",str_out)

    # Corner Cases for alignment algorithm -- edits made inside "[UNK]" tokens
    #str_out = regex.sub(r"\(log\u00b2 log n\)","(l-g\u00b2 log n)",str_out)
    #str_out = regex.sub(r"native people\u3161and a","native people - & a",str_out)
    #str_out = regex.sub(r"Greek: Αντιπατρος Βʹ ο Μακεδων", "Greek: Αντιπατρας Βʹ ο Μακεδων",str_out)
    #str_out = regex.sub(r"fragment are ΚΣ \(\"Kurios\", Lord\) and ΧΡΣ\) \(\"Christos\", Christ\)","fragment are κσ (\"Kurios\", Lord) and χρσ) (\"Christos\", Christ)",str_out)
    
    # Lower Case
    str_out = str_out.lower()

    return str_out, map_out_to_in

def reverse_map(d_list):

    # Initialize Mapping
    prev_d = 0
    new_list = []
    
    for d in d_list:
        diff = d - prev_d
        # Expand Indexes, where input mapping indicates a shift forward
        # Compress Indexes (by omission), when shifted backward; i.e. else: pass
        if diff >= 0:
            for diff_step in range(1,diff + 1):
                new_list.append(prev_d + diff_step  )
            new_list.append(prev_d + diff )
        prev_d = d

    # Reverse Direction
    new_list = [ -1*d for d in new_list ]

    return new_list

# Create Output File

with open(args.output_file, 'w') as file_out:
    writer = csv.writer(file_out,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(ft_list)

# Main Loop

with open(args.example_file, 'r') as file_in:
    for line in file_in.readlines():

        # Load Example
        input_dict = json.loads(line)
        ft_dict = {ft:0 for ft in ft_list}
        try:
            for sent in ['sent_a','sent_b']:

                dwp_tokens = input_dict[sent]['tokens'][:-1]
                dwp_masks = input_dict[sent]['mask'][:-1]
                if sent == 'sent_a':
                    dwp_tokens = dwp_tokens[1:]
                    dwp_masks = dwp_masks[1:]

                text_str = input_dict[sent]['text']
                trunc_len = input_dict[sent]['trunc']

                clean_str, map_clean_to_text = remove_accents(text_str)
                map_text_to_clean = reverse_map(map_clean_to_text)

                # Pass to BookNLP
                with open(temp_fn, 'w') as temp_txt:
                    print(text_str, file=temp_txt)
                
                booknlp.process(temp_fn, temp_dir, 'temp')

                tok_df = pd.read_csv(temp_dir+'temp.tokens', sep='\t', quoting=csv.QUOTE_NONE, dtype={'word':str}, keep_default_na=False)
                sst_df = pd.read_csv(temp_dir+'temp.supersense', sep='\t', quoting=csv.QUOTE_NONE, dtype={'text':str}, keep_default_na=False)

                # Pass to SpaCy
                parsed_str = nlp(text_str)

                # Collect Outputs
                #clean_char_index = 0
                trunc_ws,char_i = 0,0
                while char_i - trunc_ws < trunc_len:
                    if clean_str[char_i].isspace():
                        trunc_ws += 1
                    char_i += 1
                clean_char_index = char_i

                for dwp_i,dwp_tok in enumerate(dwp_tokens):

                    # Handle UNK by looking ahead to next known token
                    if dwp_tok == "[UNK]":

                        if dwp_tokens[dwp_i - 1] == "[UNK]":
                            continue

                        next_i = dwp_i
                        while next_i < len(dwp_tokens) and dwp_tokens[next_i] == "[UNK]":
                            next_i += 1

                        if next_i < len(dwp_tokens):
                            unk_char_len = clean_str[clean_char_index:].index(dwp_tokens[next_i])
                        else:
                            unk_char_len = len(clean_str[clean_char_index:])

                        dwp_tok = "".join(clean_str[clean_char_index:clean_char_index+unk_char_len].split())
                    
                    elif dwp_tok.startswith('##'):
                        dwp_tok = dwp_tok[2:]

                    # Collect NLP Features
                    dwp_pos_set, dwp_sst_set, dwp_mor_set = set(), set(), set()

                    for char in dwp_tok:

                        # Match each character in token to original string
                        while clean_str[clean_char_index].isspace():
                            clean_char_index += 1
                        
                        assert char == clean_str[clean_char_index]

                        if dwp_masks[dwp_i] == 1:

                            text_char_index = clean_char_index + map_clean_to_text[clean_char_index]

                            # BNLP Token
                            bnlp_token_index = tok_df[(tok_df['byte_onset']<=text_char_index) & (tok_df['byte_offset']>text_char_index)]['token_ID_within_document'].values[0]

                            # Check BookNLP Alignment
                            bnlp_token,bnlp_sindex,bnlp_eindex = tok_df[tok_df['token_ID_within_document']==bnlp_token_index][['word','byte_onset','byte_offset']].values[0]
                    
                            bnlp_clean_token,_ = remove_accents(bnlp_token)
                            bnlp_clean_sindex = bnlp_sindex + map_text_to_clean[bnlp_sindex]
                            bnlp_clean_eindex = bnlp_eindex + map_text_to_clean[bnlp_eindex]
                            
                            assert bnlp_token == text_str[bnlp_sindex:bnlp_eindex]
                            if not any([text_str.startswith(s) for s in corner_case_list]):
                                assert bnlp_clean_token == clean_str[bnlp_clean_sindex:bnlp_clean_eindex]
                            assert clean_char_index >= bnlp_clean_sindex and clean_char_index <= bnlp_clean_eindex

                            # Collect BookNLP Features
                            dwp_pos = tok_df[tok_df['token_ID_within_document']==bnlp_token_index]['POS_tag'].values[0]
                            dwp_pos_set.add((dwp_pos,bnlp_token_index))
                            
                            dwp_sst = sst_df[(sst_df['start_token'] <= bnlp_token_index) & (sst_df['end_token'] >= bnlp_token_index)]['supersense_category'].values
                            if dwp_sst.size > 0:
                                dwp_sst_set.add((dwp_sst[0],bnlp_token_index))
                            
                            # SpaCy Token
                            for spacy_token in parsed_str:
                                spacy_sindex,spacy_eindex = spacy_token.idx, spacy_token.idx + len(spacy_token)
                                if text_char_index >= spacy_sindex and text_char_index < spacy_eindex:
                                    break
                            
                            # Check SpaCy Alignment
                            spacy_clean_token,_ = remove_accents(spacy_token.text)
                            spacy_clean_sindex = spacy_sindex + map_text_to_clean[spacy_sindex]
                            spacy_clean_eindex = spacy_eindex + map_text_to_clean[spacy_eindex]

                            assert spacy_token.text == text_str[spacy_sindex:spacy_eindex]
                            if not any([text_str.startswith(s) for s in corner_case_list]):
                                assert spacy_clean_token == clean_str[spacy_clean_sindex:spacy_clean_eindex]
                            assert clean_char_index >= spacy_clean_sindex and clean_char_index <= spacy_clean_eindex

                            # Collect SpaCy Features
                            if spacy_token.has_morph():
                                for dwp_mor in iter(spacy_token.morph):
                                    dwp_mor_set.add((dwp_mor, spacy_token.i))
                    
                        clean_char_index += 1
                    
                    for tag_,index_ in dwp_pos_set.union(dwp_sst_set).union(dwp_mor_set):
                        ft_dict[tag_] += 1
            
        except:
            with open(args.error_file, 'a') as file_out:
                print(line, file=file_out)
            
            for ft in ft_list:
                ft_dict[ft] = -1

        # Save Feature Counts
        csv_row = [ft_dict[ft] for ft in ft_list]
        
        with open(args.output_file, 'a') as file_out:
            writer = csv.writer(file_out,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(csv_row)
