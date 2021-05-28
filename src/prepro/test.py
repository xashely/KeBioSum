import json
import re
with open('/Users/User/Documents/test6.json','r') as f:
    json_dict = json.load(f)

title = json_dict['metadata']['title']
text = ''

for p in json_dict['body_text']:
    p_text = p['text']
    print(p_text,'\n')
    if p['section'] == 'Pre-publication history':
        continue
    # remove references and citations from text
    citations = [*p['cite_spans'],*p['ref_spans']]
    if len(citations):
        for citation in citations:
            cite_span_len = citation['end']-citation['start']
            cite_span_replace = ' '*cite_span_len
            p_text  = p_text[:citation['start']] + cite_span_replace  + p_text[citation['end']:]
    # do other cleaning of text
    p_text = p_text.strip()
    p_text = re.sub('\[[\d\s,]+?\]', '', p_text) # matches references e.g. [12]
    p_text = re.sub('\(Table \d+?\)', '', p_text) # matches table references e.g. (Table 1)
    p_text = re.sub('\(Fig. \d+?\)', '', p_text) # matches fig references e.g. (Fig. 1)
    p_text = re.sub('(?<=[0-9]),(?=[0-9])', '', p_text) # matches numbers seperated by commas
    p_text = re.sub('[^\x00-\x7f]+',r'', p_text) # strips non ascii
    p_text = re.sub('[\<\>]',r' ', p_text) # strips  <> tokens which are not compatable StanfordNLPtokenizer
    p_text = re.sub('\n',' ',p_text) # replaces line break with full stop
    p_text = re.sub('\r',' ',p_text) # replaces line break with full stop
    p_text = re.sub(' +',' ',p_text) # removes multipe blank spaces. 
    print(p_text,'\n\n\n')
    text += '{:s}\n'.format(p_text)
    