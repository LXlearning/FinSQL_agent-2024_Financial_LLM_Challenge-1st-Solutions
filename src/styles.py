import os
from typing import Optional


def load_css_file(css_file_path: str) -> str:
    """
    加载CSS文件内容
    
    Args:
        css_file_path: CSS文件路径
        
    Returns:
        CSS文件内容字符串，包装在<style>标签中
    """
    try:
        # 获取当前脚本的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建CSS文件的绝对路径
        css_path = os.path.join(os.path.dirname(current_dir), css_file_path)
        
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
        
        return f"<style>\n{css_content}\n</style>"
    
    except FileNotFoundError:
        print(f"警告: CSS文件 {css_file_path} 未找到")
        return "<style>/* CSS文件未找到 */</style>"
    except Exception as e:
        print(f"警告: 加载CSS文件时出错: {e}")
        return "<style>/* CSS加载错误 */</style>"

def get_gradio_styles() -> str:
    """
    获取Gradio界面的CSS样式
    
    Returns:
        完整的CSS样式字符串
    """
    return load_css_file("static/gradio_styles.css")

# 可选：如果需要合并多个CSS文件
def load_multiple_css_files(*css_files: str) -> str:
    """
    加载并合并多个CSS文件
    
    Args:
        *css_files: CSS文件路径列表
        
    Returns:
        合并后的CSS内容
    """
    combined_css = ""
    for css_file in css_files:
        css_content = load_css_file(css_file)
        # 移除外层的<style>标签，只保留CSS内容
        css_content = css_content.replace("<style>\n", "").replace("\n</style>", "")
        combined_css += css_content + "\n"
    
    return f"<style>\n{combined_css}\n</style>" 