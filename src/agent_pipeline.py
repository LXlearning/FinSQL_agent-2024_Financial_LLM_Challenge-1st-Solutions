import argparse
import json
import multiprocessing
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from agents import Decomposer, Extractor, Selector
from const_cn import *
from LLM import LLM, init_log_path
from utils import load_json_file, load_jsonl_file, remove_commas

warnings.filterwarnings("ignore")

class AnswerQlist:
    def __init__(self, output_path, input_file, data_path, LLM_ensemble=True):
        self.output_path = output_path
        self.log_path = f'{output_path}/llm.log'
        self.output_file = f'{output_path}/output.json'
        self.data = load_json_file(input_file)

        df_table = pd.read_parquet(f'{data_path}/df_table.parquet')
        df_col = pd.read_parquet(f'{data_path}/df_col.parquet')
        df_codev = pd.read_parquet(f'{data_path}/df_code_v_desc.parquet')
        df_code_v = pd.read_parquet(f'{data_path}/df_code_v完整码值信息.parquet')

        knowledge_base = load_json_file(f"{data_path}/knowledge_base.json")
        schema_emb_list = pd.read_parquet(f'{data_path}/schema_embeddings.parquet')
        cols_emb_list = pd.read_parquet(f'{data_path}/table_columns_embeddings.parquet')
        codev_emb_list = pd.read_parquet(f'{data_path}/codev_emb_list.parquet')

        memory_path = f"{data_path}/memory.json"
        with open(memory_path, "r") as f:
            memory_list = json.load(f)

        print('预处理数据:', df_table.shape, df_col.shape, df_codev.shape, df_code_v.shape)
        self.chat_group = [Extractor(),
                           Selector(df_table, df_col, df_codev, knowledge_base,
                                    schema_emb_list, cols_emb_list, codev_emb_list),
                           Decomposer(df_col, df_code_v, memory_list, LLM_ensemble=LLM_ensemble)]

        # 断点继续功能
        all_ids = []
        for items in self.data:
            all_ids += [x['id']  for x in items['team']]
        finished_ids = set()
        if os.path.exists(self.output_file):
            output_data_lst = load_jsonl_file(self.output_file)
            for o in output_data_lst:
                finished_ids.add(o['id'])
        self.unfinished_ids = [n for n in all_ids if n not in finished_ids]
        print('unfinished_ids:', len(self.unfinished_ids))

    def run_batch(self, q_idxs):
        init_log_path(self.log_path)
        data = [self.data[i] for i in q_idxs]
        with open(self.output_file, 'a+', encoding='utf-8') as fp:
            for idx, question_list in tqdm(enumerate(data), total=len(data)):
                use_agent = 1
                history = []
                user_message_multi = []
                for q_item in question_list['team']:
                    id = q_item['id']
                    question = q_item['question']
                    # if id not in self.unfinished_ids: continue
                    user_message = {"idx": idx,
                                    "id": id, 
                                    "question": question,
                                    "history": history,
                                    "sql": "",
                                    "answer": "",
                                    "error": "",
                                    "df_desc": pd.DataFrame(),
                                    "rewritten_question": "",
                                    "history_conversation": []
                                    }
                    print('-'*20)
                    print('q:', q_item['question'])
                    if len(user_message['history']) > 0:
                        if user_message['history'][-1]['answer'] == '':
                            print('history回答失败，跳过回答')
                            use_agent = 0
                    try:
                        # if use_agent == 1:
                        for agent in self.chat_group:
                            start_time = time.time()
                            agent.talk(user_message)
                            elapsed_time = time.time() - start_time
                            print(f"Agent {agent.name} 用时: {int(elapsed_time)} 秒")

                    except Exception as e:
                        print(f"Exception: {e}, sleep 20 seconds.")
                        user_message['error'] = str(e)
                        time.sleep(3)

                    if user_message['error'] == 'Invalid authentication credentials':
                        raise Exception('SQL API证书过期!!!')

                    history.append({'question': user_message['question'], 
                                    'rewritten_question': user_message['rewritten_question'],
                                    'sql': user_message['sql'],
                                    'answer': user_message['answer'], 
                                    'df_desc': user_message['df_desc'],})
                
                    save_path1 = f'{self.output_path}/conversation'
                    if not os.path.exists(save_path1):
                        os.makedirs(save_path1)
                    pd.DataFrame(user_message['history_conversation']).to_excel(f'{save_path1}/{id}.xlsx', index=False)

                    user_message_save = {k: user_message[k] for k in ['idx', 'id', 'question','answer', 'sql', 'error', 'rewritten_question']}
                    for k, v in user_message_save.items():
                        print(f"{k} ->: {v}")
                    user_message_multi.append(user_message_save)
                    print(json.dumps(user_message_save, ensure_ascii=False), file=fp, flush=True)


def get_output_data(output_path):
    output_data_lst = load_jsonl_file(output_path)
    df = pd.DataFrame(output_data_lst)
    df['answer_new'] = df['answer'].apply(remove_commas)

    print('all:', len(df))
    num_no_error = len(df[df['error']==''])
    print('未报错:', num_no_error)
    df_cred_error = df[df['error'] == 'Invalid authentication credentials']
    print('令牌错误:', len(df_cred_error))
    print('数字单位校正:', len(df[df['answer']!=df['answer_new']]))
    df.drop('answer', axis=1, inplace=True)
    return df


def ensemble_answers(data_list):
    data_ori = data_list[0]
    data_new = data_list[1]
    for row1, row2 in zip(data_ori, data_new):
        for row1_item, row2_item in zip(row1['team'], row2['team']):
            print('-'*5, row1_item['id'], '-'*5)
            assert row1_item['id'] == row2_item['id']
            prompt = prompt_ensemble_template.format(question=row1_item['question'],
                                                     answer1=row1_item['answer'],
                                                     answer2=row2_item['answer'])
            resp = LLM(prompt)
            print(row1_item['answer'], row2_item['answer'])
            row1_item['answer'] = resp
            print('[finally ans]:', resp)
    return data_ori
