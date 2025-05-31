import json
import os
import re

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from func_timeout import func_set_timeout

load_dotenv(verbose=True, override=True)
TEAM_TOKEN = os.getenv('TEAM_TOKEN')
print('TEAM_TOKEN:', TEAM_TOKEN)


from func_timeout import func_set_timeout


# @func_set_timeout(5)
def execute_sql(sql, limit=1000):
    url = "https://comm.chatglm.cn/finglm2/api/query"
    data = {
        "sql": sql,
        "limit": limit
    }
    sql_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TEAM_TOKEN}"
    }
    for attempt_count in range(3):
        try:
            response = requests.post(url, headers=sql_headers, json=data, timeout=90)
            res = response.json()
            break
        except Exception as e:
            print(f"尝试查询sql出现异常: {e}，正在进行第{attempt_count + 1}次尝试...")
    if attempt_count >= 2:
        print('尝试3次查询sql失败, 返回null')
        return {
                    "sql": str(sql),
                    "data": [],
                    "error": '运行超时',
                }
    if res.get('success'):
        return {
                    "sql": str(sql),
                    "data": res['data'],
                    "error": "",
                }
    else:
        if res.get('detail') == 'Invalid authentication credentials':
            raise Exception('SQL API证书过期!!!')
        else:
            return {
                        "sql": str(sql),
                        "data": [],
                        "error": res['detail'],
                    }


class QueryBySql:
    def __init__(self, df1):
        # 实体包含公司名称，公司英文名，证券代码，概念名称
        # 公司中文名
        cols_use = ['table_name_all', 'column_name', 'column_description', '库_表名']
        self.df_comp_Abbr = df1[df1['column_name'].isin(['AShareAbbr', 'SecuAbbr', 'ChiNameAbbr', 'ChiName']) &
                            (df1['table_name_all'].str.contains('ConstantDB'))][cols_use].sort_values('table_name_all')
        # 英文简称
        self.df_comp_Eng = df1[df1['column_name'].isin(['ChiSpelling', 'EngName']) &
                           (df1['table_name_all'].str.contains('ConstantDB'))].sort_values('table_name')
        # 证券代码
        self.df_stock_code = df1[df1['column_name'].isin(['SecuCode']) &
                             (df1['table_name_all'].str.contains('ConstantDB'))].sort_values('table_name')
        # 概念名称
        self.df_cocept = df1[df1['column_description'].str.contains('概念名称')].sort_values('table_name')

    def find_ner_data_columns(self, resp_NER):
        if len(resp_NER) == 0:
            return {}, ""
        table_value_info_all = []
        extra_infos = []
        for k, words in resp_NER.items():
            for word in words:
                if k == '公司名称':
                    df_match = self.df_comp_Abbr
                elif k == '公司英文名':
                    df_match = self.df_comp_Eng
                elif k == '证券代码':
                    df_match = self.df_stock_code
                elif k == '概念名称':
                    df_match = self.df_cocept
                else:
                    df_match = None
                table_value_info = search_keywords_in_tables(word, df_match)
                if len(table_value_info) > 0:
                    # 只链接到一张表中
                    info_tmp = table_value_info[0]
                    # for info_tmp in table_value_info:
                    tmp = "表'{}',字段'{}'".format(info_tmp['table_name_all'], info_tmp['column_name'])
                    extra_info = f'字段值: {word} 对应关系:' + tmp
                else:
                    extra_info = ''
                    
                table_value_info_all += table_value_info
                extra_infos.append(extra_info)
        return table_value_info_all, extra_infos


def split_sql_statements(sql_string):
    """多sql拆分"""
    # 使用正则表达式来匹配 SQL 语句，假设 SQL 语句以分号结束，并且考虑可能的注释
    pattern = re.compile(r'((?:--[^\n]*\n)*[^;]+;)', re.DOTALL)
    matches = pattern.findall(sql_string)
    result = []
    for match in matches:
        # 去除注释
        clean_match = re.sub(r'--[^\n]*\n', '', match).strip()
        result.append(clean_match)
    return result


def search_keywords_in_tables(keyword, df_match):
    table_value_info = []
    for _, row in df_match.iterrows():
        table = row['table_name_all']
        column = row['column_name']
        sql = f"""SELECT {column}
FROM {table}
WHERE {column} = '{keyword}'
"""
    #like '%{keyword}%'
        sql_exec = execute_sql(sql)
        if (len(sql_exec['data']) > 0) & (not isinstance(sql_exec['data'], str)):
            table_value_info.append({"table_name_all":table, 'column_name': column})
    return table_value_info


def get_schema_desc_str(df):
    """根据召回数据库生成数据库描述信息"""
    df.fillna('' ,inplace=True)
    tables_data = {}
    current_table = None
    for (table_name, table_name_cn), df_agg in df.groupby(['table_name_all', '库_表名']):
        for _, row, in df_agg.iterrows():
            column_name = row['column_name']
            column_description = row['column_description']
            null_ratio = row['Null_Rate_str']
            if row['额外注释'] != '':
                column_notes = row['额外注释']
            else:
                column_notes = row['code_doc_pairs']
            if f'{table_name}: {table_name_cn}' != current_table:
                current_table = f'{table_name}: {table_name_cn}'
                tables_data[current_table] = []
            tables_data[current_table].append((column_name, column_description, column_notes, null_ratio))

    output_text = ""
    for table_name, columns_info in tables_data.items():
        if table_name == 'AStockMarketQuotesDB.QT_StockPerformance: 上市公司股票行情.股票行情表现(新)':
            extra_table_info = " - 收录股票从最近一个交易日往前追溯一段时期的行情表现信息，包括近1周、1周以来、近1月、1月以来、近3月、近半年、近1年、今年以来、上市以来的表现情况(以及是统计信息，时间取DATE(TradingDay) = 'XXXX-XX-XX'即可), 以及β、α、波动率、夏普比率等风险指标"
        else:
            extra_table_info = ""
        output_text += f"# Table: {table_name}{extra_table_info}\n"
        for col_info in columns_info:
            output_text += f"  [{col_info[0]}, {col_info[1]}, {col_info[2]}] {col_info[3]}"
            output_text += "\n"

    return output_text


def get_table_desc_str(df_table_candi):
    table_info = ""
    for schema_name, df_agg in df_table_candi.groupby('库名中文'):
        table_info += f"# 数据库: {schema_name}\n"
        for i, row in df_agg.iterrows():
            table_info += f"- 表名:{row['表中文']} 描述:{row['表描述简介']}\n"
            # table_info += ','.join(df_agg2['column_description'].tolist()) + '\n'
            # table_info += f"- 表名：[{row.表中文}] \n{row.表描述}\n\n"
        table_info += '\n'

    return table_info


def process_items(resp_NER):
    res_list = []
    expr_infos = []
    for key, words in resp_NER.items():
        for word in words:
            if key in ["公司名称", "公司英文名"]:
                res_lst = process_company_name(word)
                if len(res_lst) == 0:
                    print('尝试模糊匹配公司名...')
                    res_lst = process_company_name(word, fuzzy=True)
            elif key == "证券代码":
                res_lst = process_code(word)
            elif key == '概念名称':
                res_lst, expr_info = process_concept_name(word)
                if len(res_lst) == 0:
                    print('尝试模糊匹配概念...')
                    res_lst, expr_info = process_concept_name(word, fuzzy=True)
                expr_infos.append(expr_info)
            elif key == '基金名称': 
                res_lst = process_fund_name(word)
                if len(res_lst) == 0:
                    print('基金名称匹配失败, 尝试模糊匹配公司名...')
                    res_lst = process_company_name(word, fuzzy=True)
            # elif key == '行业名称':
            #     res_lst = process_industry_name(word)
            #     if len(res_lst) == 0:
            #         print('基金名称匹配失败, 尝试模糊匹配公司名...')
            #         res_lst = process_industry_name(word, fuzzy=True)
            else:
                res_lst = []
                print(f"无法识别的键：{key}, {word}")
            # 尝试再次识别
            if (len(res_lst) == 0) & (key != '基金名称'):
                res_lst = process_fund_name(word)
            res_list.extend(res_lst)

                
    # Filter out empty results
    res_list = [i for i in res_list if i]
    res = ''
    res_dict = {}
    for result_data, table_name in res_list:
        if table_name not in res_dict:
            res_dict[table_name] = result_data
        else:
            res_dict[table_name] += result_data
    for table_name, result_data in res_dict.items():
        res += f"通过表格：{table_name} 查询到以下内容, 请参考这些信息写后续sql：\n {json.dumps(result_data, ensure_ascii=False, indent=1)} \n"
        if table_name == 'AStockIndustryDB.LC_ConceptList':
            res += '\n'.join(expr_infos)
    tables = res_dict.keys()
    return res, tables


def process_fund_name(value):

    res_lst = []
    tables = ['PublicFundDB.MF_InvestAdvisorOutline', 'PublicFundDB.MF_FundProdName', 'AStockShareholderDB.LC_LegalDistribution']
    for table in tables:
        value = value.replace("'", "''")
        if table == 'PublicFundDB.MF_InvestAdvisorOutline':
            local_select_cols = ['InvestAdvisorCode', 'InvestAdvisorName', 'InvestAdvisorAbbrName']
            value1 = value.replace("公司", "")
            match_conditions = [f"InvestAdvisorName = '{value}'", f"InvestAdvisorAbbrName = '{value1}'"]
            where_clause = ' OR '.join(match_conditions)
        elif table ==  'PublicFundDB.MF_FundProdName':
            local_select_cols = ['DisclName', 'InnerCode']
            where_clause = f"DisclName = '{value}'"
        else:
            local_select_cols = ['AquirerName']
            match_conditions = [f"AquirerName = '{value}'"]
            where_clause = ' OR '.join(match_conditions)
        # 
        sql = f"""
        SELECT {', '.join(local_select_cols)}
        FROM {table}
        WHERE {where_clause}
        """
        sql_exec = execute_sql(sql)
        if (len(sql_exec['data']) > 0) & (not isinstance(sql_exec['data'], str)):
            num_max = min(len(sql_exec['data']), 2)
            res_lst.append((sql_exec['data'][:num_max], table))
    else:
        if not res_lst:
            print(f"未在任何基金表中找到代码为 {value} 的信息。")

    return res_lst


def process_industry_name(value, fuzzy=False):
    res_lst = []
    table = 'AStockIndustryDB.LC_ExgIndustry'
    local_select_cols = ['FirstIndustryName','SecondIndustryName', 'ThirdIndustryName']
    local_match_cols = ['FirstIndustryName','SecondIndustryName', 'ThirdIndustryName']
    value = value.replace("'", "''")  # Escape single quotes

    for match_col in local_match_cols:
        if fuzzy:
            where_clause = f"{match_col} LIKE '%{value}%'"
        else:
            where_clause = f"{match_col} = '{value}'"
        sql = f"""
        SELECT {', '.join(local_select_cols)}
        FROM {table}
        WHERE {where_clause}
        limit 10
        """
        sql_exec = execute_sql(sql)
        if (len(sql_exec['data']) > 0) & (not isinstance(sql_exec['data'], str)):
            res_lst.append((sql_exec['data'][:2], table))
    else:
        if not res_lst:
            print(f"未在任何表中找到代码为 {value} 的信息。")

    return res_lst


def process_concept_name(value, fuzzy=False):
    res_lst = []
    table = 'AStockIndustryDB.LC_ConceptList'
    local_select_cols = ['ClassCode', 'ClassName','SubclassCode', 'SubclassName', 'ConceptName']#'ConceptCode',
    local_match_cols = ['ClassName', 'SubclassName', 'ConceptName']
    value = value.replace("'", "''")  # Escape single quotes
    expr_info = ""
    for match_col in local_match_cols:
        if fuzzy:
            where_clause = f"{match_col} LIKE '%{value}%'"
        else:
            where_clause = f"{match_col} = '{value}'"
        sql = f"""
        SELECT {', '.join(local_select_cols)}
        FROM {table}
        WHERE {where_clause}
        """
        sql_exec = execute_sql(sql)
        if (len(sql_exec['data']) > 0) & (not isinstance(sql_exec['data'], str)):
            df_tmp = pd.DataFrame(sql_exec['data'])
            if match_col == 'ClassName':
                num_subclass = df_tmp['SubclassName'].nunique()
                num_concept = df_tmp['ConceptName'].nunique()
                expr_info = f"{value}属于'1级概念名称'ClassName, 包含'2级概念名称'SubclassName(子类){num_subclass}个和'概念名称'ConceptName{num_concept}个，以上是两条范例数据"
            elif match_col == 'SubclassName':
                num_concept = df_tmp['ConceptName'].nunique()
                expr_info = f"{value}属于'2级概念名称'SubclassName, 包含'概念名称'ConceptName(子类概念){num_concept}个，以上是两条范例数据"
            else:
                sub_name = df_tmp['SubclassName'].iloc[0]
                class_name = df_tmp['ClassName'].iloc[0]
                expr_info = f"{value}属于'概念名称'ConceptName, 上级概念: 所属2级概念名称SubclassName为{sub_name}, 所属1级概念名称ClassName为{class_name}"
            res_lst.append((sql_exec['data'][:2], table))
    else:
        if not res_lst:
            print(f"未在任何表中找到代码为 {value} 的信息。")

    return res_lst, expr_info


def process_code(value):
    """Given a code (e.g., a stock code), search the three tables and return matches.
    """
    res_lst = []
    tables = ['ConstantDB.SecuMain', 'ConstantDB.HK_SecuMain', 'ConstantDB.US_SecuMain']
    columns_to_select = ['InnerCode', 'CompanyCode', 'SecuCode', 'ChiName', 'ChiNameAbbr',
                         'EngName', 'EngNameAbbr', 'SecuAbbr', 'ChiSpelling']

    value = value.replace("'", "''")  # Escape single quotes

    for table in tables:
        local_select_cols = columns_to_select.copy()
        if 'US' in table:
            if 'ChiNameAbbr' in local_select_cols:
                local_select_cols.remove('ChiNameAbbr')
            if 'EngNameAbbr' in local_select_cols:
                local_select_cols.remove('EngNameAbbr')

        sql = f"""
        SELECT {', '.join(local_select_cols)}
        FROM {table}
        WHERE SecuCode = '{value}' OR InnerCode = '{value}'
        """
        sql_exec = execute_sql(sql)
        if (len(sql_exec['data']) > 0) & (not isinstance(sql_exec['data'], str)):
            res_lst.append((sql_exec['data'], table))
    else:
        if not res_lst:
            print(f"未在任何表中找到代码为 {value} 的信息。")

    return res_lst

def process_company_name(value, fuzzy=False):
    """
    Given a company name (or related keyword), search in three tables:
    ConstantDB.SecuMain, ConstantDB.HK_SecuMain, ConstantDB.US_SecuMain.

    Attempts to match various company-related fields (e.g., ChiName, EngName, etc.)
    and returns all matching results along with the table where they were found.
    """
    res_lst = []
    tables = ['ConstantDB.SecuMain', 'ConstantDB.HK_SecuMain', 'ConstantDB.US_SecuMain', 'USStockDB.US_CompanyInfo', 'IndexDB.LC_IndexBasicInfo']
    columns_to_match = ['CompanyCode', 'SecuCode', 'ChiName', 'ChiNameAbbr',
                        'EngName', 'EngNameAbbr', 'SecuAbbr', 'ChiSpelling']
    columns_to_select = ['InnerCode', 'CompanyCode', 'SecuCode', 'ChiName', 'ChiNameAbbr',
                         'EngName', 'EngNameAbbr', 'SecuAbbr', 'ChiSpelling']

    # Escape single quotes to prevent SQL injection
    value = value.replace("'", "''")

    for table in tables:
        # For the US table, remove columns that may not be available
        local_match_cols = columns_to_match.copy()
        local_select_cols = columns_to_select.copy()
        if 'US' in table:
            if table == 'USStockDB.US_CompanyInfo':
                local_match_cols = ['EngName', 'EngNameAbbr', 'ChiName']
                local_select_cols = ['CompanyCode', 'EngName', 'EngNameAbbr', 'ChiName']
            if table == 'ConstantDB.US_SecuMain':
                local_match_cols.remove('EngNameAbbr')
                local_select_cols.remove('EngNameAbbr')
                local_match_cols.remove('ChiNameAbbr')
                local_select_cols.remove('ChiNameAbbr')

        if table == 'IndexDB.LC_IndexBasicInfo':
            local_match_cols = ['PubOrgName']
            local_select_cols = ['IndexCode', 'PubOrgName']
        # Build the WHERE clause with OR conditions for each column
        if fuzzy:
            if value.endswith('A') and len(value) > 4:
                value = value[:-1]
            match_conditions = [f"{col} LIKE '%{value}%'" for col in local_match_cols]
        else:
            match_conditions = [f"{col} = '{value}'" for col in local_match_cols]
        where_clause = ' OR '.join(match_conditions)

        sql = f"""
        SELECT {', '.join(local_select_cols)}
        FROM {table}
        WHERE {where_clause}
        """
        sql_exec = execute_sql(sql)
        if (len(sql_exec['data']) > 0) & (not isinstance(sql_exec['data'], str)):
            res_lst.append((sql_exec['data'], table))
    else:
        # The 'else' clause in a for loop runs only if no 'break' was encountered.
        # Here it just prints if no results were found.
        if not res_lst:
            print(f"未在任何表中找到公司名称为 {value} 的信息。")

    return res_lst


def replace_date_with_day(sql):
    """
    This function replaces instances of exact date conditions in a SQL 
    statement from a format like:
        TradingDate = 'YYYY-MM-DD'
    to:
        date(TradingDate) = 'YYYY-MM-DD'
    
    Parameters:
        sql (str): The original SQL statement.
        
    Returns:
        str: The modified SQL statement, or the original if no match is found.
    """
    # Regex pattern to match patterns like: ColumnName = 'YYYY-MM-DD'
    pattern = r"([.\w]+)\s*=\s*'(\d{4}-\d{2}-\d{2})'"
    pattern2 = r"([.\w]+)\s+BETWEEN\s*'(\d{4}-\d{2}-\d{2})'\s+AND\s*'(\d{4}-\d{2}-\d{2})'"
    def replace_func(match):
        column_name = match.group(1)
        date_value = match.group(2)
        return f"date({column_name}) = '{date_value}'"
    def replace_func2(match):
        column_name = match.group(1)
        start_date = match.group(2)
        end_date = match.group(3)
        return f"date({column_name}) BETWEEN '{start_date}' AND '{end_date}'"

    
    new_sql = re.sub(pattern, replace_func, sql)
    new_sql = re.sub(pattern2, replace_func2, new_sql)

    # If no change was made, return the original SQL
    return new_sql if new_sql != sql else sql

if __name__ == "__main__":
    sql = """SELECT count(*)
FROM AStockOperationsDB.LC_Staff A
"""
    df = execute_sql(sql)
    print(df)
    print('测试finish')