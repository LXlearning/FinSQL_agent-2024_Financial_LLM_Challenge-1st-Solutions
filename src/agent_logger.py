import io
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List


class TeeOutput:
    """同时输出到多个流的工具类"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


class AgentLogger:
    """Agent处理过程日志捕获器"""
    
    def __init__(self):
        self.logs = {
            "extractor": [],
            "selector": [], 
            "decomposer": []
        }
        self.current_agent = None
        self.step_logs = []
    
    @contextmanager
    def capture_agent_output_with_console(self, agent_name: str):
        """捕获指定agent的输出，同时保持终端显示"""
        self.current_agent = agent_name.lower()
        self.step_logs = []
        
        # 保存原始stdout
        old_stdout = sys.stdout
        string_io = io.StringIO()
        
        # 创建同时输出到终端和字符串流的Tee对象
        sys.stdout = TeeOutput(old_stdout, string_io)
        
        try:
            yield self
        finally:
            # 获取捕获的输出
            captured_output = string_io.getvalue()
            sys.stdout = old_stdout
            
            # 解析并存储输出
            self._parse_agent_output(captured_output)
    
    @contextmanager
    def capture_agent_output(self, agent_name: str):
        """捕获指定agent的输出（原始方法，保持兼容性）"""
        self.current_agent = agent_name.lower()
        self.step_logs = []
        
        # 保存原始stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            yield self
        finally:
            # 获取捕获的输出
            captured_output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            # 解析并存储输出
            self._parse_agent_output(captured_output)
    
    def _parse_agent_output(self, output: str):
        """解析agent输出并分类存储"""
        lines = output.strip().split('\n')
        
        for line in lines:
            if line.strip():
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = {
                    "timestamp": timestamp,
                    "content": line.strip(),
                    "type": self._classify_log_type(line)
                }
                
                if self.current_agent in self.logs:
                    self.logs[self.current_agent].append(log_entry)
                self.step_logs.append(log_entry)
    
    def _classify_log_type(self, line: str) -> str:
        """分类日志类型"""
        line_lower = line.lower()
        
        if '-----' in line:
            return "section_header"
        elif '关键词:' in line or 'keywords' in line_lower:
            return "keywords"
        elif '召回' in line or 'recall' in line_lower:
            return "recall"
        elif 'sql' in line_lower:
            return "sql"
        elif 'memory' in line_lower or '记忆' in line:
            return "memory"
        elif 'error' in line_lower or '错误' in line or '失败' in line:
            return "error"
        elif '港股' in line or '美股' in line or 'A股' in line:
            return "stock_type"
        elif '用时:' in line or 'elapsed' in line_lower:
            return "timing"
        elif 'thought' in line_lower or '思考' in line:
            return "reasoning"
        else:
            return "info"
    
    def get_agent_logs(self, agent_name: str) -> List[Dict]:
        """获取指定agent的日志"""
        return self.logs.get(agent_name.lower(), [])
    
    def get_latest_step_logs(self) -> List[Dict]:
        """获取最新步骤的日志"""
        return self.step_logs
    
    def clear_logs(self):
        """清空所有日志"""
        for agent in self.logs:
            self.logs[agent] = []
        self.step_logs = [] 