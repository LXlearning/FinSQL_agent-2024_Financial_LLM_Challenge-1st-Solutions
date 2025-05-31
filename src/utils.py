import json
import re


def parse_sql_from_string(input_string):
    sql_pattern = r"[`']{3}sql(.*)[`']{3}"
    all_sqls = []
    # 将所有匹配到的都打印出来
    for match in re.finditer(sql_pattern, input_string, re.DOTALL):
        all_sqls.append(match.group(1).strip())
    
    if all_sqls:
        res = all_sqls[-1]
    else:
        res = input_string
    res = res.replace('<|EOT|>', '').replace('```', '').replace('sql:', '')
    return res


def load_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        print(f"load json file from {path}")
        return json.load(f)


def load_jsonl_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            js_str = line.strip()
            if js_str == '':
                continue
            js = json.loads(js_str)
            data.append(js)
        print(f"load jsonl file from {path}")
        return data
 

def add_info_to_year(date_str):
    year_pattern = r"(\d{4}年)"
    result = re.sub(year_pattern, lambda x: x.group(1) + "(信息发布日期)", date_str)
    return result


def check_year_pattern(text):
    year_pattern = r"(\d+.*?年)"
    date_pattern = r"(\d{4}-\d{2}-\d{2})"
    if re.search(year_pattern, text):
        return True
    if re.search(date_pattern, text):
        return True
    for key_word in ['半年', '年度', '季度', '月份', '时间', '时候', '天', '自然日']:
        if key_word in text:
            return True
    return False

def check_statistic_features(question):
    features = ["近一季度", "近一月", "近一个月",
                "近一周", "近三月", "近一年", "近六个月", "近三个月"]
    matched_features = []
    for feature in features:
        if feature in question:
            matched_features.append(feature)
    return matched_features


def get_sql_error(text):
    re1 = re.compile(r"\(\w+\):\s*(.*?)\[SQL",re.DOTALL)
    match = re1.search(text)
    if match:
        return match.groups()[0]
    else:
        return ''


def eval_test(x):
    try:
        res = eval(x)
    except:
        res = {}
    return res


def add_backticks_to_matches(question, columns):
    """在文本中匹配列名，并在匹配到的列名前后添加``符号。"""
    columns = [x for x in columns if len(x) >= 5]
    cols_all_sorted = sorted(columns, key=len, reverse=True)
    pattern = '|'.join(map(re.escape, cols_all_sorted))
    matched_question = re.sub(pattern, r'`\g<0>`', question)#\g<0> 表示匹配到的整个正则表达式的子组
    return matched_question


def get_code_value(text):
    """获取码值信息"""
    re1 = re.compile('(?:具体描述|具体标准|以下常量)：(.*)', re.DOTALL)
    match = re1.search(text)
    if match:
        return match.groups()[0]
    else:
        return ''
    

def get_code_value_yesno(text):
    """获取码值信息"""
    re1 = re.compile('(?:以下常量|具体描述)[：。](.*)', re.DOTALL)
    match = re1.search(text)
    if match:
        return match.groups()[0]
    else:
        return ''


def code_value2dict(text):
    """将码值信息转换为字典"""
    # re1= re.compile(r"[。.)]+$")
    # text = re.sub(re1, "", text)     # 使用 sub 函数将匹配到的部分替换为空字符串
    pattern = re.compile(r"(\d+)-([^，。)]+)")#

    result = []
    matches = pattern.findall(text)
    for match in matches:
        key = match[0]
        value = match[1]
        result.append({'code':key, 'doc': value})

    if len(result) == 0:
        pattern = re.compile(r"([A-Z0-9]+)[ —]+([^、。]+)")
        matches = pattern.findall(text)
        for match in matches:
            key = match[0]
            value = match[1]
            result.append({'code':key, 'doc': value})
    return result


def combine_code_doc(group):
    """将 code 和 doc 组合成 code-doc 对"""
    code_doc_pairs = []
    for index, row in group.iterrows():
        if is_pure_number(row['code']):
            code_doc_pairs.append(f"{row['code']}-{row['doc']}")
        else:
            code_doc_pairs.append(f"{row['doc']}")
    return "值:(" + ','.join(code_doc_pairs) + ")"


def is_pure_number(string):
    pattern = re.compile(r'^\d+$')
    if pattern.match(string):
        return True
    else:
        return False


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def remove_commas(text):
    """使用正则表达式的sub函数将找到的数字中的逗号去掉"""
    pattern = r'\d{1,3}(,\d{3})*(\.\d+)?'
    result = re.sub(pattern, lambda x: x.group().replace(',', ''), text)
    return result


def extract_data_for_desp(text):
    # 定义正则表达式模式
    pattern = r'(补充|介绍|记录|收录|存储|反映|境外投资者|包括|公司自有资金|1\.)(.*?)(?:\n|。|$)'
    # 使用 findall 方法查找所有匹配项
    matches = re.findall(pattern, text)
    result_list = [match[0]+match[1].strip() for match in matches]
    # 将列表中的元素用换行符连接成一个字符串
    result_str = '\n'.join(result_list)
    return result_str


def get_stock_source(rewritten_question, ChiNameAbbr_list):
    if '港股' in rewritten_question and not any(abbr in rewritten_question for abbr in ChiNameAbbr_list):
        return '港股'
    elif ('美股' in rewritten_question or 'CN' in rewritten_question) and not any(abbr in rewritten_question for abbr in ChiNameAbbr_list):
        return '美股'
    return 'A股'

def parse_query_result(text, end_marker="请回答，查询结果有多个时不要省略"):
    """使用正则表达式解析【查询结果】和指定结束标记之间的内容
    """
    pattern = r'【查询结果】(.*?)' + re.escape(end_marker)
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return ""


if __name__ == "__main__":
    text = """1.从技术职称、专业、文化程度、年龄等几个方面介绍公司职工构成情况。
2.数据范围：1999-12-31至今
3.信息来源：定期报告、招股说明书等"""
    extract_data_for_desp(text)