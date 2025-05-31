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

from agent_pipeline import AnswerQlist, ensemble_answers, get_output_data
from config import DATA_PATH, DEBUG, INPUT_FILE, OUTPUT_DIR, RUN_TIMES, LLM_ensemble
from LLM import LLM, init_log_path
from utils import load_json_file, load_jsonl_file

warnings.filterwarnings("ignore")


def process_single_run(run_time, now):
    output_name = now.strftime("%Y%m%d_%H%M%S")
    output_path = f'{OUTPUT_DIR}/{output_name}_{run_time}'
    init_log_path(f'{output_path}/llm.log')
    
    print('start!!!', output_path)
    run_answer = AnswerQlist(output_path, INPUT_FILE, DATA_PATH, LLM_ensemble=LLM_ensemble)
    idsx = [0, 1] if DEBUG else list(range(len(run_answer.data)))
    run_answer.run_batch(idsx)
    print('predict finished!!!')

    # 保存结果
    df1 = get_output_data(os.path.join(output_path, 'output.json'))
    no_data_num = 0
    data = load_json_file(INPUT_FILE)
    for question_list in tqdm(data):
        qa_new = []
        for q_item in question_list['team']:
            df_tmp = df1[df1['id'] == q_item['id']]
            if len(df_tmp) > 0:
                q_item['answer'] = df_tmp['answer_new'].iloc[0]
            else:
                no_data_num += 1
                q_item['answer'] = ''
            qa_new.append(q_item)
        question_list['team'] = qa_new
    print('<not f num>:', no_data_num)
    print('[data]***', len(data))
    return data


if __name__ == '__main__':
    now = datetime.now()
    data_list = []
    for run_time in range(RUN_TIMES):
        data = process_single_run(run_time, now)
        data_list.append(data)

    final_data = data_list[0] if RUN_TIMES <= 1 else ensemble_answers(data_list)

    save_path = f'{OUTPUT_DIR}/answer.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    print('-'*5, 'ensemble finished', '-'*5)
    print('finished:', save_path)
    