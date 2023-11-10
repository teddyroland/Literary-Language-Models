import csv
import json
import argparse
import pandas as pd

# I/O Arguments

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--parse_file', type=str, required=True)
arg_parser.add_argument('--wiki_model_loss_file', type=str, required=True)
arg_parser.add_argument('--full_model_loss_file', type=str, required=True)
arg_parser.add_argument('--output_file', type=str, required=True)

args = arg_parser.parse_args()

# Load Example Loss & NLP Features

wiki_loss_df = pd.read_csv(args.wiki_model_loss_file)
full_loss_df = pd.read_csv(args.full_model_loss_file)

data_parse_df = pd.read_csv(args.parse_file)

# Remove Processing Errors (>0.1% total) & Re-align Data

parse_index = data_parse_df[data_parse_df['ADJ']!=-1].index.tolist()

wiki_loss_df = wiki_loss_df.loc[parse_index].reset_index(drop=True)
full_loss_df = full_loss_df.loc[parse_index].reset_index(drop=True)
data_parse_df = data_parse_df.loc[parse_index].reset_index(drop=True)

# Tag List

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
            'VerbForm=Part', 'VerbType=Mod']

all_tags = pos_tags + sst_tags + mor_tags

# Compute Expected Loss

def expected_loss(parse_df, loss_df):

    expected_loss_dict = {}

    mlm_count_levels = list(loss_df['MLM_count'].unique())

    for tag in all_tags:
        tag_sum, tag_ct = 0, 0
            
        for mlm_ct in mlm_count_levels:
            
            sub_index = loss_df[loss_df['MLM_count']==mlm_ct].index.tolist()
            sub_df = parse_df.loc[sub_index]
            
            tag_count_levels = sub_df.groupby(tag).count()['_']
            
            for dex in tag_count_levels.index[:-1]:
                if dex+1 not in tag_count_levels.index:
                    continue
                count_dex = sub_df[sub_df[tag]==dex].index.tolist()
                greater_dex = sub_df[sub_df[tag]==dex+1].index.tolist()

                count_loss = loss_df.loc[count_dex,'Loss'].values.mean()
                greater_loss = loss_df.loc[greater_dex,'Loss'].values.mean()
                diff_loss = greater_loss - count_loss

                tag_sum += diff_loss * len(count_dex)
                tag_ct += len(count_dex)

        expected_loss_dict[tag] = tag_sum / tag_ct if tag_ct > 0 else float('nan')

    return expected_loss_dict


wiki_loss_dict = expected_loss(data_parse_df, wiki_loss_df)
full_loss_dict = expected_loss(data_parse_df, full_loss_df)

# Collect Results

tag_type_list = [['pos',pos_tags],['sst',sst_tags],['mor',mor_tags]]
row_list = []

for tag_type, tag_list in tag_type_list:
    for tag in tag_list:

        doc_count = (data_parse_df[tag] > 0).sum()

        row = [tag, tag_type, doc_count, wiki_loss_dict[tag], full_loss_dict[tag]]
        row_list.append(row)

col_list = ['Tag', 'Tag Type', 'Doc Count','Expected-Loss-Per-Word (Wiki)', 'Expected-Loss-Per-Word (Full)']
tag_loss = pd.DataFrame(row_list, columns=col_list)
tag_loss.to_csv(args.output_file, index=False)
