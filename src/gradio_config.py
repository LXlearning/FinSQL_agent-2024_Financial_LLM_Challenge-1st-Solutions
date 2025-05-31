"""
Gradio应用配置文件
"""

# Gradio应用配置
GRADIO_CONFIG = {
    "title": "Fin SQL Agent",
    "description": "金融领域NL2SQL智能助手，支持多轮对话交互，精准拆解复杂金融问题",
    "server_name": "127.0.0.1",
    "server_port": 7860,
    "share": True,
    "debug": True,
    "show_error": True,
    "theme": "soft",  # 可选: "soft", "default", "monochrome"
}

# 示例查询模板
EXAMPLE_QUERIES = [
    "600872的全称、A股简称、法人、法律顾问、会计师事务所及董秘是？",
    "该公司实控人是否发生改变？如果发生变化，什么时候变成了谁？是哪国人？是否有永久境外居留权？",
    "在实控人发生变化的当年股权发生了几次转让？",
    "今天是2021年12月24日，创近半年新高的股票有几只？",
    "哪些股票股价大于75，且同时当天创一年新高的是？",
    "以上股票连续两日（今日与昨日）满足上述要求的是？"
]