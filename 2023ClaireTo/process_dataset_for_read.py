import re
import json
import pandas as pd
import os
df_persona_input = pd.read_excel('/Users/tzhang/projects/A2I2/2023ClaireTo/Data/compiled_annotations_proposal.xlsx')
df_persona = df_persona_input[['civilian','character_role']]
persona_dict = dict(zip(df_persona["civilian"], df_persona["character_role"]))
directory = "/Users/tzhang/projects/A2I2/2023ClaireTo/Dialogues/"

results = []
turn = 0
for file in os.listdir(directory):
    result = {}
    name = file.split('_')[2]
    persona = persona_dict[name]
    df_temp = pd.read_excel(directory + file)
    utterances = []
    for i in range(len(df_temp)):
        speaker = df_temp['speaker'][i]
        utterance = df_temp['text'][i]
        if 'You have some residents across the main thoroughfare there that are refusing to evacuate at this' in utterance:
            print(file)
            import pdb; pdb.set_trace()
        if speaker == 1:
            utterances.append(['Q',utterance])
        else:
            utterances.append([name,utterance])
            #context += name + ': ' + utterance
        turn+=1
    #import pdb; pdb.set_trace()
    last_index = max(j for j, sub in enumerate(utterances) if sub[0] == name)
    question = ''
    for j,sub in enumerate(utterances[:last_index]):
        #import pdb; pdb.set_trace()
        question+=sub[0]+ ': ' + sub[1]+'*****'
    answer = utterances[last_index][1]
    result = {
            "question": f"persona: {name + ' -- a ' + persona +'.'} context: {question}",
            "answers": [answer]
        }
    results.append(result)
    
    
print(turn)
#1111 turns #104 data entries
converted_data_100 = results[:80]

output_file = "/Users/tzhang/projects/A2I2/data_for_train/predict_data_80.jsonl"
with open(output_file, "w") as f:
    for entry in converted_data_100:
        f.write(json.dumps(entry) + "\n")
import pdb; pdb.set_trace()