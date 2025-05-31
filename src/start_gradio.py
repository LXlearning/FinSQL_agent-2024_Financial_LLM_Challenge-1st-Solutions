#!/usr/bin/env python3
"""
NL2SQL Gradio应用启动脚本
"""

import argparse
import logging
import os
import signal
import sys
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gradio_app import create_interface
from gradio_config import GRADIO_CONFIG


def setup_logging():
    """设置日志"""
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = f"{log_dir}/gradio_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def signal_handler(signum, frame):
    """信号处理"""
    logger.info(f"接收到信号 {signum}，正在关闭应用...")
    sys.exit(0)

def check_dependencies():
    """检查依赖"""
    required_modules = [
        'gradio', 'pandas', 'numpy', 'tqdm'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"缺少必要依赖: {', '.join(missing_modules)}")
        logger.error("请运行: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动NL2SQL Gradio应用")
    parser.add_argument("--host", default=GRADIO_CONFIG["server_name"], help="服务器地址")
    parser.add_argument("--port", type=int, default=GRADIO_CONFIG["server_port"], help="端口号")
    parser.add_argument("--share", action="store_true", default=GRADIO_CONFIG["share"], help="是否分享链接")
    parser.add_argument("--debug", action="store_true", default=GRADIO_CONFIG["debug"], help="调试模式")
    
    args = parser.parse_args()
    
    # 设置日志
    global logger
    logger = setup_logging()
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=" * 50)
    logger.info("NL2SQL Gradio应用启动")
    logger.info("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查必要目录
    required_dirs = ["../output", "../logs", "../data"]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logger.warning(f"目录不存在，正在创建: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    try:
        # 创建并启动应用
        logger.info("正在初始化Gradio界面...")
        demo = create_interface()
        
        logger.info(f"启动服务器: {args.host}:{args.port}")
        logger.info(f"调试模式: {'开启' if args.debug else '关闭'}")
        logger.info(f"分享链接: {'开启' if args.share else '关闭'}")
        
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭应用...")
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("应用已关闭")

if __name__ == "__main__":
    main() 