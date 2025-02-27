import json
import pandas as pd
import random
import re



answer = '/Users/tzhang/projects/A2I2/results/answer_80.jsonl'
input_data = '/Users/tzhang/projects/A2I2/data_for_train/predict_data_80.jsonl'

question_lst = []
with open(input_data,'r') as f:
    for line in f:
        line = json.loads(line)
        question_lst.append(line['question'])
#import pdb; pdb.set_trace()


answer_lst = []
with open(answer, 'r') as f:   
    for line in f:
        line = json.loads(line)
        #import pdb; pdb.set_trace()
        index = line['index']
        idx = int(index)
        #import pdb; pdb.set_trace()
        question = question_lst[idx]
        response = str(line['response'])
        answer = str(line['answers']).strip('[]')
        answer_lst.append([index,question,response,answer])
#import pdb; pdb.set_trace()            
df = pd.DataFrame(answer_lst, columns=['index_number','persona+context','model_pred','answer'])

all_persona = []
all_utters=[]
all_ans = []
all_model = []
all_index_number=[]
for i in range(len(df)):
    index_number = df['index_number'][i]
    ans = [df['answer'][i]]
    model_pred = [df['model_pred'][i]]
    data = df['persona+context'][i]
    persona_part = data.split("context:")[0].strip()
    pattern = r"persona:\s*(\w+)"
    match = re.search(pattern, persona_part)    
    name = match.group(1)
    persona_lines = persona_part.replace("persona:","").strip().split(".")
    persona=[line.strip() for line in persona_lines if line.strip()]

    # Extract the context and split into Q and R utterances
    context_part = data.split("context:")[1].strip()
    lines = context_part.split("*****")
    
    utterances=[]
    # Extract Q's and R's utterances
    cnt = 0
    for line in lines:
        if 'Q:' in line:
            utterances.append([cnt,line,''])
        if str(name+':') in line:
            utterances.append([cnt,'',line])
        cnt+=1   
    #import pdb; pdb.set_trace()
    diff = len(utterances) - len(persona)

    if diff >= 0:
        persona.extend(['']*diff)
    else:
        print('the number of diff is negative')
        diff = 2
        #import pdb;pdb.set_trace()
    num = len(utterances) - 1
    ans_lst = ['']*num + ans
    model_lst = ['']*num + model_pred
    index_number_lst = [index_number] * len(utterances)
    #import pdb; pdb.set_trace()
    all_index_number.append(index_number_lst)
    all_ans.append(ans_lst)
    all_model.append(model_lst)
    all_persona.append(persona)
    all_utters.append(utterances)


dataframes=[]
for i in range(len(all_index_number)):
    try:
        df_temp = pd.DataFrame({
            'index_number': all_index_number[i],
            'Persona': all_persona[i],
            'utt_cnt': [sublist[0] for sublist in all_utters[i]],
            'Q': [sublist[1] for sublist in all_utters[i]],
            'R': [sublist[2] for sublist in all_utters[i]],
            'model_pred':all_model[i],
            'answer': all_ans[i],
        })
        dataframes.append(df_temp)
    except:
        continue
final_df = pd.concat(dataframes, ignore_index=True)
final_df.to_excel('/Users/tzhang/Documents/A2I2_data/readable_result_0210.xlsx')
import pdb; pdb.set_trace()

def shuffle_row_columns(row):
    column_order = df_ans.columns.tolist()
    shuffled_order = column_order.copy()
    random.shuffle(shuffled_order)  # Shuffle column order
    
    # Reorder row based on shuffled column order
    shuffled_row = row[shuffled_order]
    return pd.Series([shuffled_row.tolist(), shuffled_order])

df_ans = final_df[['answer','baseline_pred','lapdog_pred']]
shuffled_results = df_ans.apply(shuffle_row_columns,axis=1)
df_head=final_df[['index_number','Persona','Q','R']]


# Extract results and rebuild the shuffled DataFrame
shuffled_df = pd.DataFrame(shuffled_results[0].tolist(), columns=df_ans.columns)
shuffled_df['column_order'] = shuffled_results[1].tolist()
df_head['generation1']=shuffled_df['answer']
df_head['generation2']=shuffled_df['baseline_pred']
df_head['generation3']=shuffled_df['lapdog_pred']
df_head['column_order'] = shuffled_df['column_order']
df_head.to_excel('/Users/tzhang/Documents/lapdog_shuffled_result_new_with_order.xlsx')
import pdb; pdb.set_trace()
df_all.to_excel('/Users/tzhang/Documents/lapdog_new.xlsx')