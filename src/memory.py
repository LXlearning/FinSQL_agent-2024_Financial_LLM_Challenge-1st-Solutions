import json

MEMORY_PROMPT = """
你是一个高级排序器代理。
请从给定的候选查询列表中精心挑选出与当前问题最为相似、问题中含有的关键元素最类似、解决路径最为吻合的前2个查询。
随后，按照相似度从高到低的顺序，将这些查询的问题序号以列表形式清晰排列。确保最贴近的查询位于列表的顶部。
请注意，不需要选公司名称最相似的，而是要选问题结构最相似的。

问题：{question}

候选列表：
{query_list}


问题：{question}
请按照以下json格式进行输出，可以被Python json.loads函数解析。不回答问题，不作任何解释，不输出其他任何信息。
```json
{{
    "相关问题": 
}}
``` 
"""


def get_memory(Memory_idxs, memory_list):
    memory = []
    memory_q = []
    for memory_id in Memory_idxs:
        one_query = memory_list[memory_id-1]["rewritten_question"]
        one_solution = memory_list[memory_id-1]["SQL"]
        memory_q.append(one_query)
        memory.append(f"""问题: {one_query}
【回答】
{one_solution}
""")
    memory = '\n'.join(memory)
    memory_q = '\n'.join(memory_q)
    return memory, memory_q


def match_knowledge_base(question, knowledge_base):
    way_list = []
    for keyword in knowledge_base:
        if any([x in question for x in keyword.split('|')]):
            way_list.append(knowledge_base[keyword])
    return way_list

if __name__ == "__main__":
    question = "年报中，公司是否披露了财务报表审计报告?"
    file_path = './devlop_home/data/interim/knowledge_base.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        knowledge_base =  json.load(f)
    result = match_knowledge_base(question, knowledge_base)
    print(result)