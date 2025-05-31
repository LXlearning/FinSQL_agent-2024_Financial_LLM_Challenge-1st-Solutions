import asyncio
import re

import cpca
import numpy as np
import pandas as pd

from agent_tools import tools_codev
from const_cn import *
from embedding import cal_score, find_similar_docs, get_top_k
from LLM import (
    LLM,
    LLM_get_json_response,
    concurrent_llm_calls,
    get_embeddings,
    prase_json_from_response,
)
from memory import MEMORY_PROMPT, get_memory, match_knowledge_base
from sql_retrieve import (
    execute_sql,
    get_schema_desc_str,
    get_table_desc_str,
    process_items,
    replace_date_with_day,
)
from table_utils import find_primary_key_relationships
from utils import *


class BaseAgent():
    def __init__(self):
        pass

    def talk(self, message: dict):
        pass


class Extractor(BaseAgent):
    """query解析模块"""

    def __init__(self):
        super().__init__()
        self.name = "Query Extractor"

    def talk(self, message: dict):
        print('-----Extractor-----')
        question = message.get('question')
        history_info = parse_history(message['history'], type=2)
        if len(message['history']) == 0:
            rewritten_question = question
        else:
            prompt = rewrite_q_template.format(context_text=history_info, question=question)
            rewritten_question = LLM(prompt)

        if '跳空低开' in rewritten_question:
            rewritten_question = rewritten_question.replace('跳空低开', '跳空低开(当日开盘价<昨日最低价)')
        if "PBX" in question:
            rewritten_question = rewritten_question.replace('PBX', 'PBX(用收盘价加权计算)')
            print('rewritten_question:', rewritten_question)
        prompt = key_word_template1.format(question=rewritten_question)
        resp_keywords1 = LLM_get_json_response(prompt)
        if not resp_keywords1:
            print('关键词抽取失败,尝试重新抽取')
            his1 = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": resp_keywords1}]
            resp_keywords1 = LLM_get_json_response('关键词抽取失败,尝试重新抽取', history=his1)

        prompt = key_word_template2.format(question=rewritten_question)
        resp_keywords2 = LLM_get_json_response(prompt)

        message['rewritten_question'] = rewritten_question
        message['keywords1'] = resp_keywords1
        message['keywords2'] = resp_keywords2
        print('关键词:', resp_keywords1, resp_keywords2)
        return 

class Selector():
    """Schema Linking模块"""
    
    def __init__(self, df_table, df_col, df_codev, knowledge_base, 
                 schema_emb_list, cols_emb_list, codev_emb_list):
        super().__init__()
        self.name = "Schema Linking"
        self.common_cols = ['ID', 'JSID', '公司代码', '发布时间', '更新时间', '信息发布日期', '信息来源', '修改日期',
       'No description available', '证券内部编码', '日期', '与上市公司关联关系', '首次信息发布日期',
       '事件进程', '货币单位', '序号', '备注']
        self.df_table = df_table
        df_col = df_col.drop(df_col[(df_col['column_name'] == 'EndDate') & (df_col['table_name'] == 'LC_ConceptList')].index)
        self.df_col = df_col
        self.df_codev = df_codev
        self.knowledge_base = knowledge_base

        # 内键
        inner_key_cols = ['InnerCode', 'CompanyCode', 
                          'ConceptCode','InvestAdvisorCode',"IndustryNum"]
        self.inner_key_desc = list(df_col[df_col['column_name'].isin(inner_key_cols)]['column_description'].unique())
        inner_keys = df_col[df_col['column_name'].isin(inner_key_cols)].groupby('column_name')['table_name'].apply(list)
        # print('内键数量:',  inner_keys.apply(len))
        self.inner_keys = inner_keys.to_dict()

        # 列表名embedding
        self.schema_emb_list = [tuple(x.values()) for x in schema_emb_list.to_dict('records')]
        self.cols_emb_list = [tuple(x.values()) for x in cols_emb_list.to_dict('records')]
        self.cols_emb_list = list({t[0]: t for t in self.cols_emb_list}.values()) # 去重
        self.codev_emb_list = [tuple(x.values()) for x in codev_emb_list.to_dict('records')]
        sql_special_name = """SELECT CHiNameAbbr
FROM ConstantDB.SecuMain
WHERE CHiNameAbbr like '%港股%'
OR CHiNameAbbr like '%香港%';
"""
        self.CHiName_list = [x['CHiNameAbbr'] for x in execute_sql(sql_special_name)['data']]

    def talk(self, message):
        """ 1. 列名召回 & 码值召回 & 表格召回
            2. 根据召回的表格和列名做初筛，LLM做表格选择
            3. LLM & embedding融合
            4. 证券主表选择
            5. 相似度高的表列 + 实体抽取的表列 + 日期判断列 + 外键 + 历史表列
        """
        print('-----Selector-----')
        question = message.get('question')
        rewritten_question = message['rewritten_question']
        keywords1 = message.get('keywords1')
        if "股价" in question:
            keywords1['关键名词'].append("最高价")
        if not isinstance(keywords1, dict):
            print('关键词解析有问题:', keywords1)
            keywords1 = {"关键名词": [], "时间": []}
        keywords2 = message.get('keywords2')
        history = message.get('history')
        if not isinstance(keywords2, dict):
            print('关键词解析有问题:', keywords2)
            keywords2 = {}
        extra_info, ner_table = process_items(keywords2)

        # 判断公司数据来源, 优先证券主表，其次港股美股
        schema_HK = ['港股数据库.港股公司概况', '港股数据库.港股公司员工数量变动表', '港股数据库.港股行情表现', '常量库.港股证券主表']
        schema_US = ['常量库.美股证券主表', '美股数据库.美股公司概况', '美股数据库.美股日行情']
        if ("借贷" in rewritten_question) | ("A股" in rewritten_question):
            SecuMain_name = '证券主表'
            df_col4select = self.df_col[~(self.df_col['库_表名'].isin(schema_HK+schema_US))]
        elif (('ConstantDB.HK_SecuMain' in ner_table) & ('ConstantDB.SecuMain' not in ner_table)) | (get_stock_source(rewritten_question, self.CHiName_list) == '港股'):
            SecuMain_name = '港股证券主表'
            df_col4select = self.df_col[(self.df_col['库_表名'].isin(schema_HK))]
            print('港股数据库查询')
        elif (('ConstantDB.US_SecuMain' in ner_table) & ('ConstantDB.SecuMain' not in ner_table))  | (get_stock_source(rewritten_question, self.CHiName_list) == '美股'):
            SecuMain_name = '美股证券主表'
            df_col4select = self.df_col[(self.df_col['库_表名'].isin(schema_US))]
            print('美股数据库查询')
        else:
            SecuMain_name = '证券主表'
            df_col4select = self.df_col[~(self.df_col['库_表名'].isin(schema_HK+schema_US))]
        
        # 1. embeeding召回
        schema_embs = [x for x in self.schema_emb_list if x[0] in df_col4select['库_表名'].tolist()]
        columns_embs = [x for x in self.cols_emb_list if x[0] in df_col4select['column_description'].tolist()]
        question_emb_list = get_embeddings([question])
        doc_emb_list = get_embeddings(keywords1.get('关键名词', [question]))

        ## 表召回
        df_recall_schema = find_similar_docs(doc_emb_list, schema_embs, score_threshold=0.3, topk=2)
        ## 列召回: 关键词召回&问题召回
        df_recall_cols1 = find_similar_docs(doc_emb_list, columns_embs, score_threshold=0.4, topk=40)
        df_recall_cols1 = df_recall_cols1[~df_recall_cols1['doc'].isin(self.common_cols)]
        df_recall_cols2 = find_similar_docs(question_emb_list, columns_embs, score_threshold=0.3, topk=8)
        df_recall_cols2 = df_recall_cols2[~df_recall_cols2['doc'].isin(self.common_cols)]

        df_recall_cols = pd.concat([df_recall_cols1, df_recall_cols2]).drop_duplicates('doc')
        recall_cols = list(set(df_recall_cols['doc'].tolist()))
        print('召回字段num: ', len(recall_cols))
        ## 码值召回
        df_recall_codev = find_similar_docs(doc_emb_list, self.codev_emb_list, score_threshold=0.4, topk=15)
        df_recall_codev = df_recall_codev.merge(self.df_codev, on='doc', how='left')
        if len(df_recall_codev) > 0:
            df_recall_codev = df_recall_codev.groupby(['column_description', 'table_name']).apply(combine_code_doc).reset_index(name='code_doc_pairs')
        else:
            df_recall_codev = pd.DataFrame(columns=['column_description', 'table_name', 'code_doc_pairs'])
        recall_cols += list(df_recall_codev['column_description'].unique())
        recall_cols += self.inner_key_desc  # 主键字段也加到召回列中

        # 2.选择表格(LLM+recall+排名靠前的列召回)
        df_table_candi = self.df_table[self.df_table['表中文'].isin(df_col4select['表中文'])]
        table_info = get_table_desc_str(df_table_candi)
        prompt = selector_template.format(question=question, table_info=table_info)
        resp_select = LLM_get_json_response(prompt)
        if isinstance(resp_select, dict):
            table_LLM = resp_select.get('名称', []) if isinstance(resp_select.get('名称'), list) else []
        else:
            table_LLM = []
        table_recall = [x.split('.')[-1] for x in df_recall_schema['doc']]

        col_best = df_recall_cols1[df_recall_cols1['score']>0.49].head(5)['doc'].tolist()
        table_recall_in_col = list(df_col4select[(df_col4select['column_description'].isin(col_best))]['表中文'].unique())

        print('LLM选择表:', table_LLM)
        print('emb选择表:', table_recall)
        print('emb col选择表:', table_recall_in_col)
        table_select = table_LLM + table_recall + table_recall_in_col

        # 3.规则列名,证券主表选择
        recall_cols += ['中文名称','中文名称缩写', '证券简称', '获配企业名称', '行业名称', '是否合并', '是否执行']
        if ('证券主表' not in table_select):
            print('LLM未选择证券主表')
            table_select.append(SecuMain_name)
        conditions_cols = {"调整": "是否调整", "CN": "国家"}
        recall_cols.extend([conditions_cols[key] for key in conditions_cols if key in question])
        df = cpca.transform([question])
        if df[['省', '市', '区']].iloc[0].isnull().sum() < 3:
            recall_cols += ['公司注册地址']

        # 4. 集成表格选择: 表列合并 + 实体抽取的表列 + 日期判断 + 外键
        # 历史字段
        df_desc = df_col4select[(df_col4select['column_description'].isin(recall_cols)) & 
                                (df_col4select['表中文'].isin(table_select))]
        if len(history) > 0:
            df_desc_history = history[-1]['df_desc']
        else:
            df_desc_history = pd.DataFrame()
        cols = [x for x in df_desc_history.columns if x not in ['code_doc_pairs']]
        df_desc = pd.concat([df_desc, df_desc_history[cols]]).drop_duplicates()

        ## 日期字段添加
        if check_year_pattern(rewritten_question):
            df_desc_date = df_col4select[(df_col4select['库_表名'].isin(df_desc['库_表名'])) & 
                                   (df_col4select['额外注释'].str.contains('【日期字段】'))]
        else:
            df_desc_date = pd.DataFrame()
        df_desc = pd.concat([df_desc, df_desc_date]).drop_duplicates(['库_表名','column_description'])
        
        ## 拼接码值信息
        df_desc = df_desc.merge(df_recall_codev, on=['table_name', 'column_description'], how='left')
        df_desc['code_doc_pairs'].fillna('',inplace=True)
        # print('最终召回表:', df_desc['table_name_all'].value_counts())

        # 找到召回的列对应的外键信息: 查看召回的表关联需要哪些外键
        table_name_all = df_desc['table_name_all'].unique().tolist()
        table_name_all = [x.split('.')[-1] for x in table_name_all]
        inner_keys_info = find_primary_key_relationships(table_name_all, self.inner_keys)
        ## 特殊外键
        if "LC_IndexBasicInfo" in table_name_all:
            inner_keys_info.append('LC_IndexBasicInfo.IndexCode = SecuMain.InnerCode')
        if "LC_SHTypeClassifi" in table_name_all:
            inner_keys_info.append('LC_IndexBasicInfo表:,当SHAttribute=2时, SHID = SecuMain.CompanyCode; 当SHAttribute=3时, SHID = SecuMain.InnerCode')
        if "LC_ActualController" in table_name_all:
            inner_keys_info.append('LC_ActualController.ControllerCode = SecuMain.CompanyCode, 关联得到实际控制人的名称，企业性质等信息。')
            
        schema_info = get_schema_desc_str(df_desc) + '\n【表外键关联方式】' + '\n'.join(inner_keys_info)

        # 知识库召回info
        way_list = match_knowledge_base(question, self.knowledge_base)
        stat_col = check_statistic_features(question)
        if len(stat_col) > 0:
            way_list.append(f"'{','.join(stat_col)}'有相关字段来统计历史时间数据，需要用【日期字段】筛选, 通常是DATE(TradingDay) ='xxxx-xx-xx")
        if check_year_pattern(rewritten_question):
            way_list.append("问题中提到日期，请使用【日期字段】筛选")
        message['df_desc'] = df_desc
        message['extra_info'] = extra_info  + '\n'.join(way_list)
        message['schema_info'] = schema_info
        return 
    

class Decomposer(BaseAgent):
    """SQL Agent模块"""

    def __init__(self, df_col, df_code_v, memory_list, LLM_ensemble=True):
        super().__init__()
        self.name = "SQL Agent"
        common_cols = ['ID', 'JSID', '公司代码', '发布时间', '更新时间', '信息发布日期', '信息来源', '修改日期',
       'No description available', '证券内部编码', '日期', '与上市公司关联关系', '首次信息发布日期',
       '事件进程', '货币单位', '序号', '备注']
        self.df_col = df_col
        cols_all = list(df_col['column_description'].unique())
        cols_all = list(set(cols_all) - set(common_cols))
        self.cols_all = [re.sub(r"\(.*?\)", "", s) for s in cols_all]
        self.code_v_cols = list(df_code_v['column_name'].unique())
        self.LLM_ensemble = LLM_ensemble
        print('LLM_ensemble:', self.LLM_ensemble)

        self.memory_list = memory_list
        self.df_memory = pd.DataFrame(memory_list)
        mask_embeddings = get_embeddings(self.df_memory['standart_question'].to_list())
        self.df_memory['mask_embedding'] = [x[1] for x in mask_embeddings]
        self.similarity_q = SimilarityQuery(self.df_memory)
        print('记忆数量:', len(memory_list))

    def talk(self, message: dict, max_rounds=12):
        print('-----Decomposer-----')
        question, schema_info, extra_info = message['question'], message['schema_info'], message['extra_info']
        q_new = message['rewritten_question']
        # q_new = add_backticks_to_matches(message['rewritten_question'],  self.cols_all)
        # if q_new != message['rewritten_question']:
        #     print('列名增强:', q_new)
        sim_query = self.similarity_q.simily(q_new)#, score_threshold=0.85
        memory_q = [x['rewritten_question'] for x in sim_query]
        memory_scores = [x['score'] for x in sim_query]
        memory_scores_max = max(memory_scores) if len(memory_scores) > 0 else 0
        memory_qa = '\n\n'.join(['【问题】 ' + x['rewritten_question'] + '\n【回答】\n' + x['SQL'] for x in sim_query])
        print('memory q:\n', memory_q)
        if len(memory_q) > 0:
            memory_info = "以下是一些相近问题的参考示例：/n" + memory_qa + "/n参考示例结束"
        else:
            memory_info = ""
        history_info = parse_history(message['history'])
        prompt1 = decompose_template.format(schema_info=schema_info, 
                                            question=q_new,
                                            memory_info=memory_info, 
                                            extra_info=extra_info,
                                            date_info_prompt=date_info_prompt,
                                            history_info=history_info)
        
        prompt_shuffle = decompose_template2.format(schema_info=schema_info, 
                                                    question=q_new,
                                                    memory_info=memory_info, 
                                                    extra_info=extra_info,
                                                    date_info_prompt=date_info_prompt,
                                                    history_info=history_info)
        if (memory_scores_max > 0.9) | (self.LLM_ensemble is False):
            resp_sql = LLM(prompt1)
        else:
            resp_sql = self.multi_LLMs(prompt1,
                                       question=q_new,
                                       prompt_shuffle=prompt_shuffle,
                                       history=[])
        his_llm = [{"role": "user", "content": prompt1},
                   {"role": "assistant", "content": resp_sql}]  # 查询的LLM对话
        his_summrize = [{"role": "system",
                         "content": Action_Thought_template.format(question=q_new)}]  # 总结的LLM对话
        history_conversation = []  # 保存对话历史
        sql_all = []
        next_sql_flag = 1 # 执行sql判断
        next_action_flag = 0 # 任务状态判断
        
        none_data_flag = 0
        for round_count in range(max_rounds-2):
            # 执行sql 并获取结果
            if next_sql_flag == 1:
                try:
                    if  "<answer>" in resp_sql:
                        res_sql = extract_xml_answer(resp_sql)
                    res_sql = parse_sql_from_string(res_sql) # 查询结果
                    res_sql = replace_date_with_day(res_sql)
                    sql_exec_info = execute_sql(res_sql, limit=50)

                    if sql_exec_info['error'] != '': ## sql报错
                        error_info = get_sql_error(sql_exec_info['error'])
                        prompt = refiner_template.format(sqlite_error=error_info)
                    elif len(sql_exec_info['data']) == 0: ## 查不到数据
                        prompt = refiner_data_template
                        none_data_flag += 1
                    else: ## 查到数据，判断是否需要继续or总结回答
                        next_action_flag = 1
                        resp_tools = f"""【查询过程】
{resp_sql}
【查询结果】
{str(sql_exec_info['data'])}
请回答，查询结果有多个时不要省略"""
                        ### 判断查询数据是否有码值数据, 有的话直接先去查码值
                        for col, v in sql_exec_info['data'][0].items():
                            if col in self.code_v_cols:#  & (isinstance(v, int))
                                next_action_flag = 0.5
                except Exception as e: ## sql解析失败
                    res_sql = ''
                    prompt = f'回答的sql解析失败error: {str(e)}，请严格按照格式生成sql'

            # 2次查不到数，认为无数据
            if (none_data_flag > 2) & ('是否' in question):
                next_action_flag = 1
                resp_tools = f"""【查询过程】
{resp_sql}
【查询结果】
无数据
请回答，查询结果有多个时不要省略"""
            if next_action_flag == 1:
                none_data_flag = 0  # 下个action，空数据计数清零
                resp = LLM(resp_tools, history=his_summrize)
                his_summrize += [{"role": "user", "content": resp_tools}, 
                                    {"role": "assistant", "content": resp}]
                print('-'*10)
                print('thought resp:', resp)
                sql_all.append(res_sql)
                for round_repeat in range(3):
                    try:
                        decompose_response = prase_json_from_response(resp)
                        thought = decompose_response["思考"]
                        action = decompose_response["行动"]
                        prompt = f"""上一步结果不足以回答问题,请继续生成下一步的sql,思路: {thought}"""
                        if action not in Action_list:
                            prompt_action = Action_Thought_Rerty_template
                        else:
                            break
                    except Exception as e:
                        action = "[结束]"
                        prompt_action = Action_Thought_Rerty_template
                    resp = LLM(prompt_action, history=his_summrize)
                    thought = decompose_response["思考"]
                    action = decompose_response["行动"]
                    his_summrize += [{"role": "user", "content": prompt_action}, 
                                        {"role": "assistant", "content": resp}]
            elif next_action_flag == 0.5:
                action = '[码值查询]'
                thought = ''
            else:
                action = '[sql重新生成]'
                thought = ''

            # 结束时跳出循环，否则进行校正/继续回答
            if "[结束]" in action:
                thought = LLM("思考过程已经找到答案，请根据问题将答案总结出来", history=his_summrize) # 最终的thought
                break
            elif action == '[码值查询]':
                resp_tools = tools_codev(self.df_col, sql_exec_info)
                next_sql_flag = 0
                next_action_flag = 1
            elif action in ['[sql重新生成]', '[继续]']:
                if (memory_scores_max > 0.9) | (self.LLM_ensemble is False):
                    resp_sql = LLM(prompt, history=his_llm)
                else:
                    resp_sql = self.multi_LLMs(prompt,
                                               question=q_new,
                                               prompt_shuffle=prompt_shuffle,
                                               history=his_llm)
                print('-'*10)
                print('sql resp:', resp_sql)
                his_llm += [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": resp_sql}]
                next_sql_flag = 1
                next_action_flag = 0
            else:
                raise ValueError(f'action不符合预期: {action}')

        history_conversation += his_llm + his_summrize
        message['sql'] = "\n-----\n".join(sql_all)
        message['history_conversation'] = history_conversation

        # 总结
        prompt_answer = summarize_template.format(question=q_new, evidence=thought)
        new_answer = LLM(prompt_answer)
        message['answer'] = new_answer
        print('finised decompose')
        return

    def multi_LLMs(self, prompt, question, prompt_shuffle='', history=[]):
        # 准备并行查询的参数
        import time
        start = time.time()
        queries = []
        for i in range(5):
            queries.append({"query": prompt, "history": history, "pred_param": i})
        # 处理打乱顺序的prompt
        if prompt_shuffle != '':
            if len(history) == 0:
                queries.append({"query": prompt_shuffle, "history": [], "pred_param": 3})
            else: # 有history时，只修改history中的初始prompt
                history_new = history.copy()
                history_new[0] = {"role": "user", "content": prompt_shuffle}
                queries.append({"query": prompt, "history": history_new, "pred_param": 3})
        
        # 多路径候选生成
        results = asyncio.run(concurrent_llm_calls(queries))
        # print(f"并行执行时间: {int(time.time() - start)} 秒")
        
        resp_all = []
        for result in results:
            if result['success']:
                resp_sql = result['response']
                if "<answer>" in resp_sql:
                    res_sql = extract_xml_answer(resp_sql)
                res_sql = parse_sql_from_string(res_sql)
                res_sql = replace_date_with_day(res_sql)
                sql_data = execute_sql(res_sql, limit=50)['data']
                resp_all.append({'resp_sql': resp_sql, 'sql_data': sql_data})

        # 选择最优路径
        resp_all_filter = [resp for resp in resp_all if len(resp['sql_data'])>0]
        count_resp = len(resp_all_filter)
        print('可选择路径:', count_resp)
        if len(resp_all_filter) == 0:
            return resp_all[0]['resp_sql']
        else:
            resp_info = sql_ensemble_template.format(count_resp=count_resp, question=question)
            # 查询过程
            for i, resp_qa in enumerate(resp_all_filter):
                resp_info += f"""【查询过程{i}】
{resp_qa['resp_sql']}
【查询结果{i}】
{resp_qa['sql_data'][:5]}
"""
            # 选择
            resp_num = LLM(resp_info)
            # print('multi_LLMs:', resp_num)
            try:
                resp_num = int(prase_json_from_response(resp_num)['best_num'])
                if abs(resp_num) >= count_resp:
                    print('multi_LLMs选择超出范围', resp_num)
                    resp_num = 0
            except Exception as e:
                print('multi_LLMs解析有问题：', e)
                resp_num = 0
            print('multi_LLMs最终选择:', resp_num)
            return resp_all_filter[resp_num]['resp_sql']


def parse_history(history, type=1):
    result = []
    for item in history:
        if type == 1:
            formatted_str = "问: {}\nsql:\n{}".format(item["rewritten_question"], item["sql"])
        elif type == 2:
            formatted_str = "问: {}\n答: {}".format(item["rewritten_question"], item["answer"])
        result.append(formatted_str)
    history_str = "\n\n".join(result)
    return history_str


class SimilarityQuery:
    def __init__(self, df_memory):
        self.df_memory = df_memory

    def simily(self, query, score_threshold=0.8, top_k=2):
        # 获取所有的memory_embeddings
        memory_embeddings = self.df_memory['mask_embedding'].to_list()

        # 对query进行mask
        prompt = mask_prompt_template.format(question=query)
        mask_query =  LLM(prompt)
        # 对query进行embedding
        query_embedding = get_embeddings([mask_query])[0][1]
        score_list = []
        for memory_embed in memory_embeddings:
            score_list.append(cal_score(query_embedding,memory_embed))
        res_top_k = get_top_k(score_list,score_threshold,top_k)

        sim_memory = []
        if res_top_k!=[]:
            for i in res_top_k:
                dic = {}
                index = i[0]
                score = i[1]
                dic['rewritten_question'] = self.df_memory.loc[index, 'rewritten_question']
                dic['score'] = score
                dic['SQL'] = self.df_memory.loc[index, 'SQL']
                sim_memory.append(dic)
        return sim_memory

