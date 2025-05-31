import os

# 基础配置
RUN_TIMES = 1  # 问题重复运行次数(1 or 2)
DEBUG = True  # 是否开启调试模式
LLM_ensemble = True  # 生成SQL时是否开启LLM ensemble

INPUT_FILE = "./data/questions/金融复赛a榜.json"  # 输入问题文件路径
OUTPUT_DIR = "./output/复赛A榜"  # 输出目录
DATA_PATH = "./data/interim"  # 中间数据存储路径

# 确保输出目录存在
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
