import asyncio
import functools
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from zhipuai import ZhipuAI

# from openai import OpenAI

load_dotenv(verbose=True, override=True)
LLM_model_name = os.getenv('LLM_model_name')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
MAX_RETRY = 3
print('LLM_model_name:', LLM_model_name)


def init_log_path(my_log_path):
    global log_path
    log_path = my_log_path
    dir_name = os.path.dirname(log_path)
    os.makedirs(dir_name, exist_ok=True)


def get_embeddings(data: list, max_batch_size=20, tqdm_flag=False):
    embeddings_result = []
    if tqdm_flag:
        data_iter = tqdm(range(0, len(data), max_batch_size), total=len(data) // max_batch_size + (1 if len(data) % max_batch_size > 0 else 0))
    else:
        data_iter = range(0, len(data), max_batch_size)
    for index in data_iter:
        batch_data = data[index: index + max_batch_size]
        for attempt_count in range(6):
            try:
                resp = zhipu_embeddings(batch_data)
                break
            except Exception as e:
                print(f"尝试获取嵌入向量时出现异常: {e}，正在进行第{attempt_count + 1}次尝试...")
        embeddings_result += resp
    return embeddings_result


def zhipu_embeddings(batch_data, dimensions=1024):
    client = ZhipuAI()
    response = client.embeddings.create(
        model="embedding-3",
        input=batch_data,
        dimensions=dimensions
    )
    resp = [(x, y.embedding) for x, y in zip(batch_data, response.data)]
    return resp


def LLM(query, history=[], pred_param=0):
    global log_path
    if pred_param == 0:
        do_sample = False
        temperature = 0.1
        top_p = 0.8
    elif pred_param == 1:
        do_sample = True
        temperature = 0.4
        top_p = 0.95
    elif pred_param == 2:
        do_sample = True
        temperature = 0.6
        top_p = 0.99
    else:
        do_sample = True
        temperature = 0.8
        top_p = 0.99
    # 模型选择
    if 'glm' in LLM_model_name:
        client = ZhipuAI()
    # elif 'deepseek' in LLM_model_name:
    #     client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    if (log_path is None):
        raise FileExistsError('log_path is None, init_log_path first!')
    with open(log_path, 'a+', encoding='utf8') as log_fp:
        print('='*60, file=log_fp)
        print('-'*20 + 'input_prompt' +'-'*20, file=log_fp)
        print(query, file=log_fp)
        print('-'*20 + 'sys_response' +'-'*20, file=log_fp)
        messages = history + [{"role": "user", "content": query}]
        for attempt_count in range(6):
            try:
                response = client.chat.completions.create(
                    model=LLM_model_name, #glm-4-air
                    messages=messages,
                    stream=False,
                    max_tokens=8000,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    seed=2025,
                )
                break
            except Exception as e:
                print(f"尝试调用LLM出现异常: {e}，正在进行第{attempt_count + 1}次尝试...")
        resp = response.choices[0].message.content
        print(resp, file=log_fp)
        print(f'input token: {response.usage.prompt_tokens}, output token: {response.usage.completion_tokens}')
    return resp


# 并发执行器 - 限制最大并发数为8
async def concurrent_llm_calls(queries: List[Dict[str, Any]], max_workers: int = 8) -> List[Dict[str, Any]]:
    """
    并发调用LLM接口
    
    参数:
        queries: 包含查询参数的字典列表，每个字典应包含:
            - query: 用户查询内容
            - history: 对话历史(可选)
            - pred_param: 预测参数(可选)
        max_workers: 最大并发数，默认为8
    
    返回:
        包含每个查询结果的字典列表
    """
    # 创建线程池执行器
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        loop = asyncio.get_event_loop()
        # 准备所有任务
        tasks = []
        for query_data in queries:
            query = query_data["query"]
            history = query_data.get("history", [])
            pred_param = query_data.get("pred_param", 0)
            
            # 绑定参数到LLM函数
            partial_llm = functools.partial(LLM, query, history, pred_param)
            # 提交任务到线程池
            task = loop.run_in_executor(executor, partial_llm)
            tasks.append(task)
        
        # 并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        final_results = []
        for i, result in enumerate(results):
            query_data = queries[i]
            if isinstance(result, Exception):
                final_results.append({
                    "query": query_data["query"],
                    "success": False,
                    "error": str(result)
                })
            else:
                final_results.append({
                    "query": query_data["query"],
                    "success": True,
                    "response": result
                })
        
        return final_results


def prase_json_from_response(rsp: str):
    pattern = r"```json(.*?)```"
    rsp_json = None
    try:
        match = re.search(pattern, rsp, re.DOTALL)
        if match is not None:
            try:
                rsp_json = json.loads(match.group(1).strip())
            except:
                pass
        else:
            rsp_json = json.loads(rsp)
        return rsp_json
    except json.JSONDecodeError as e:  # 因为太长解析不了
        try:
            match = re.search(r"\{(.*?)\}", rsp, re.DOTALL)
            if match:
                content = "[{" + match.group(1) + "}]"
                return json.loads(content)
        except:
            pass
        print(rsp)
        raise ("Json Decode Error: {error}".format(error=e))


def LLM_get_json_response(query, max_retries=MAX_RETRY,history=[]):
    for retry in range(max_retries):
        try:
            response = LLM(query, history=history)
            if '```' in response:
                response = prase_json_from_response(response)
            else:
                response = json.loads(response)
            return response
        except Exception as e:
            print(e)
            if retry == max_retries - 1:
                return response
