import pandas as pd

from LLM import LLM, LLM_get_json_response


def tools_codev(df_col, sql_exec_info):
    """工具：解析sql中查到的码值"""
    resp = LLM_get_table_columns(sql_exec_info['sql'])
    try:
        df = df_col.merge(pd.DataFrame(resp), on=['table_name', 'column_name'])
    except:
        print(f"tools_codev 解析sql有问题{resp}")
    desc = []
    if len(df) > 0:
        for _, row in df.iterrows():
            desc.append(f"{row['table_name']}, {row['column_name']}: 注释: {row['注释']}")
    desc = '\n'.join(desc)
    codev_expr_prompt = f"""请根据表格信息，将我查询出来的sql结果根据注释的码值信息进行码值转换。
【表格】
{desc}
【查询结果】
{sql_exec_info['data']}
只给我最后转换的结果，不输出中间过程。请按照json格式进行输出
"""
    codev_resp = LLM(codev_expr_prompt)
    return codev_resp


def LLM_get_table_columns(sql):
    prompt = """请根据sql代码，提取出sql中SELECT语句选择的表名和列名，只给我最后提取的结果，不输出中间过程。请按照以下json格式进行输出
[{{
    "表名": [],
    "列名": []
}}]
案例:
-----
【sql】
SELECT  A.SecuCode, 
        A.ChiName, 
        B.ChangePCTTM
FROM ConstantDB.HK_SecuMain A
JOIN HKStockDB.CS_HKStockPerformance B 
ON A.InnerCode = B.InnerCode
WHERE  DATE(B.TradingDay) = '2020-04-29' 

【答案】
```json
[
    {{
        "table_name": "HK_SecuMain",
        "column_name": "SecuCode"
    }},
    {{
        "table_name": "HK_SecuMain",
        "column_name": "ChiName"
    }},
    {{
        "table_name": "CS_HKStockPerformance",
        "column_name": "ChangePCTTM"
    }}
]
```

【sql】
{sql}


    """
    resp = LLM_get_json_response(prompt.format(sql=sql))
    return resp
