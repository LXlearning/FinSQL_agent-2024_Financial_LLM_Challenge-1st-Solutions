import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd

from agent_logger import AgentLogger
from agent_pipeline import AnswerQlist
from config import DATA_PATH, DEBUG, LLM_ensemble
from LLM import LLM, init_log_path, prase_json_from_response
from styles import get_gradio_styles  # 导入CSS样式
from utils import extract_xml_answer, parse_query_result, parse_sql_from_string


class NL2SQLChatBot:
    def __init__(self):
        self.conversations = {}  # 存储所有对话
        self.current_conversation_id = None
        self.agent_system = None
        self.agent_logger = AgentLogger()  # 新增日志捕获器
        self.init_error = None  # 新增：保存初始化错误信息
        self.initialize_agent()
        
    def initialize_agent(self):
        """初始化Agent系统"""
        # 创建临时输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"../output/gradio_temp/{timestamp}"
        os.makedirs(output_path, exist_ok=True)
        
        # 初始化日志
        init_log_path(f'{output_path}/llm.log')
        
        # 创建虚拟输入文件用于Agent初始化
        temp_input = [{"team": [{"id": "temp", "question": "temp"}]}]
        temp_file = f"{output_path}/temp_input.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(temp_input, f, ensure_ascii=False)
        
        # 初始化AnswerQlist
        try:
            self.agent_system = AnswerQlist(
                output_path=output_path,
                input_file=temp_file,
                data_path=DATA_PATH,
                LLM_ensemble=LLM_ensemble
            )
            self.init_error = None  # 清除错误状态
            return "✅ Agent系统初始化成功"
        except Exception as e:
            self.agent_system = None  # 确保初始化失败时设为None
            error_msg = f"❌ Agent系统初始化失败: {str(e)}"
            self.init_error = error_msg  # 保存具体错误信息
            print(error_msg)
            return error_msg

    def create_new_conversation(self):
        """创建新对话"""
        conversation_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversations[conversation_id] = {
            "id": conversation_id,
            "title": "新对话",
            "messages": [],
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        self.current_conversation_id = conversation_id
        return conversation_id

    def get_conversation_list(self):
        """获取对话列表"""
        conversations = list(self.conversations.values())
        # 按更新时间排序
        conversations.sort(key=lambda x: x["updated_at"], reverse=True)
        return conversations

    def switch_conversation(self, conversation_id):
        """切换对话"""
        if conversation_id in self.conversations:
            self.current_conversation_id = conversation_id
            return self.conversations[conversation_id]["messages"]
        return []

    def update_conversation_title(self, conversation_id, first_message):
        """根据第一条消息更新对话标题"""
        if conversation_id in self.conversations:
            # 截取前20个字符作为标题
            title = first_message[:20] + "..." if len(first_message) > 20 else first_message
            self.conversations[conversation_id]["title"] = title
            self.conversations[conversation_id]["updated_at"] = datetime.now()

    def process_query_with_realtime_updates(self, user_input: str):
        """处理用户查询 - 实时更新版本"""
        if not user_input.strip():
            yield [], ""
        
        # 检查Agent系统是否已正确初始化
        if self.agent_system is None:
            # 如果Agent系统未初始化，显示错误信息
            if not self.current_conversation_id:
                self.create_new_conversation()
            
            conversation = self.conversations[self.current_conversation_id]
            messages = conversation["messages"]
            
            # 添加用户消息
            user_message = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "round": len(messages) // 2 + 1
            }
            messages.append(user_message)
            
            # 如果是第一条消息，更新对话标题
            if len(messages) == 1:
                self.update_conversation_title(self.current_conversation_id, user_input)
            
            # 添加具体的错误消息
            error_content = self.init_error or "❌ Agent系统初始化失败，无法处理查询。请检查系统配置或重启应用。"
            error_message = {
                "role": "assistant",
                "content": error_content,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "round": user_message["round"]
            }
            messages.append(error_message)
            
            # 更新对话时间
            conversation["updated_at"] = datetime.now()
            
            # 显示系统错误状态，包含具体错误信息
            error_agent_process = {
                "step1": {"status": "error", "content": f"Agent系统未正确初始化<br/>{self.init_error or ''}", "logs": []},
                "step2": {"status": "pending", "content": "", "logs": []},
                "step3": {"status": "pending", "content": "", "logs": []}
            }
            
            yield messages, format_enhanced_agent_process(error_agent_process)
            return
        
        # 如果没有当前对话，创建新对话
        if not self.current_conversation_id:
            self.create_new_conversation()
        
        conversation = self.conversations[self.current_conversation_id]
        messages = conversation["messages"]
        
        # 添加用户消息
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "round": len(messages) // 2 + 1
        }
        messages.append(user_message)
        
        # 如果是第一条消息，更新对话标题
        if len(messages) == 1:
            self.update_conversation_title(self.current_conversation_id, user_input)
        
        # 清空之前的日志
        self.agent_logger.clear_logs()
        
        # 初始化agent处理过程
        agent_process = {
            "step1": {"status": "pending", "content": "", "logs": []},
            "step2": {"status": "pending", "content": "", "logs": []},
            "step3": {"status": "pending", "content": "", "logs": []}
        }
        
        try:
            # 构建历史对话信息
            history = []
            for i in range(0, len(messages)-1, 2):
                if i+1 < len(messages)-1:
                    prev_msg = messages[i]
                    prev_reply = messages[i+1] if i+1 < len(messages)-1 else None
                    if prev_reply and prev_reply.get("role") == "assistant":
                        history.append({
                            'question': prev_msg['content'],
                            'rewritten_question': prev_msg['content'],
                            'sql': prev_reply.get('sql', ''),
                            'answer': prev_reply['content'],
                            'df_desc': pd.DataFrame(),
                        })
            
            # 准备用户消息给agent系统
            user_message_for_agent = {
                "idx": user_message["round"],
                "id": f"gradio_{self.current_conversation_id}_{user_message['round']}",
                "question": user_input,
                "history": history,
                "sql": "",
                "answer": "",
                "error": "",
                "df_desc": pd.DataFrame(),
                "rewritten_question": "",
                "history_conversation": []
            }
            
            print('-'*20)
            print('q:', user_input)
            
            # Step 1: Query Extractor
            agent_process["step1"]["status"] = "processing"
            agent_process["step1"]["content"] = "正在解析查询..."
            yield messages, format_enhanced_agent_process(agent_process)
            
            with self.agent_logger.capture_agent_output_with_console("extractor"):
                start_time = time.time()
                self.agent_system.chat_group[0].talk(user_message_for_agent)
                elapsed_time = time.time() - start_time
                print(f"Agent {self.agent_system.chat_group[0].name} 用时: {int(elapsed_time)} 秒")
            
            extractor_logs = self.agent_logger.get_latest_step_logs()
            agent_process["step1"] = {
                "status": "success",
                "content": self._format_extractor_results(user_message_for_agent, extractor_logs),
                "logs": extractor_logs
            }
            yield messages, format_enhanced_agent_process(agent_process)
            
            # Step 2: Schema Selector
            agent_process["step2"]["status"] = "processing"
            agent_process["step2"]["content"] = "正在进行Schema链接..."
            yield messages, format_enhanced_agent_process(agent_process)
            
            with self.agent_logger.capture_agent_output_with_console("selector"):
                start_time = time.time()
                self.agent_system.chat_group[1].talk(user_message_for_agent)
                elapsed_time = time.time() - start_time
                print(f"Agent {self.agent_system.chat_group[1].name} 用时: {int(elapsed_time)} 秒")
            
            selector_logs = self.agent_logger.get_latest_step_logs()
            agent_process["step2"] = {
                "status": "success", 
                "content": self._format_selector_results(user_message_for_agent, selector_logs),
                "logs": selector_logs
            }
            yield messages, format_enhanced_agent_process(agent_process)
            
            # Step 3: SQL Decomposer
            agent_process["step3"]["status"] = "processing"
            agent_process["step3"]["content"] = "正在生成SQL..."
            yield messages, format_enhanced_agent_process(agent_process)
            
            with self.agent_logger.capture_agent_output_with_console("decomposer"):
                start_time = time.time()
                self.agent_system.chat_group[2].talk(user_message_for_agent)
                elapsed_time = time.time() - start_time
                print(f"Agent {self.agent_system.chat_group[2].name} 用时: {int(elapsed_time)} 秒")
            
            decomposer_logs = self.agent_logger.get_latest_step_logs()
            agent_process["step3"] = {
                "status": "success",
                "content": self._format_decomposer_results(user_message_for_agent, decomposer_logs),
                "logs": decomposer_logs
            }
            
            # 获取最终结果
            sql_query = user_message_for_agent.get("sql", "")
            answer = user_message_for_agent.get("answer", "")
            
            # 添加系统回答
            system_message = {
                "role": "assistant",
                "content": answer or "查询处理完成，请查看AI思考过程了解详细步骤。",
                "sql": sql_query,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "round": user_message["round"]
            }
            messages.append(system_message)
            
            # 更新对话时间
            conversation["updated_at"] = datetime.now()
            
            yield messages, format_enhanced_agent_process(agent_process)
            
        except Exception as e:
            print(f"Exception: {e}")
            error_message = {
                "role": "assistant",
                "content": f"处理查询时出现错误: {str(e)}",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "round": user_message["round"]
            }
            messages.append(error_message)
            
            # 更新失败的步骤
            for step in ["step1", "step2", "step3"]:
                if agent_process[step]["status"] == "processing":
                    agent_process[step] = {
                        "status": "error",
                        "content": f"处理失败: {str(e)}",
                        "logs": []
                    }
                    break
            
            conversation["updated_at"] = datetime.now()
            yield messages, format_enhanced_agent_process(agent_process)

    def _format_extractor_results(self, message: Dict, logs: List[Dict]) -> str:
        """格式化Extractor结果"""
        rewritten_q = message.get('rewritten_question', '')
        keywords1 = message.get('keywords1', {})
        keywords2 = message.get('keywords2', {})
        
        result = f"""<div class="step-result">
        <h4>问题改写</h4>
        <p>{rewritten_q or '无需改写'}</p>
        
        <h4>关键词抽取结果</h4>
        <div class="code-block">
            <div class="code-header">JSON</div>
            <pre><code>{json.dumps(keywords1, ensure_ascii=False, indent=2) if keywords1 else '{}'}</code></pre>
        </div>
        
        <h4>实体抽取结果</h4>
        <div class="code-block">
            <div class="code-header">JSON</div>
            <pre><code>{json.dumps(keywords2, ensure_ascii=False, indent=2) if keywords2 else '{}'}</code></pre>
        </div>
        </div>"""
        
        return result

    def _format_selector_results(self, message: Dict, logs: List[Dict]) -> str:
        """格式化Selector结果"""
        df_desc = message.get('df_desc', pd.DataFrame())
        schema_info = message.get('schema_info', '')
        
        tables = ', '.join(df_desc['表中文'].unique()) if not df_desc.empty else '无'
        field_count = len(df_desc) if not df_desc.empty else 0
        
        result = f"""<div class="step-result">
        <h4>选择的数据表</h4>
        <p>{tables}</p>
        
        <h4>召回的字段数量</h4>
        <p>{field_count} 个字段</p>
        
        <h4>Schema信息</h4>
        <div class="schema-info">
            <pre>{schema_info}</pre>
        </div>
        </div>"""
        
        return result

    def _format_decomposer_results(self, message: Dict, logs: List[Dict]) -> str:
        """格式化Decomposer结果"""
        sql = message.get('sql', '')
        answer = message.get('answer', '')
        history_conversation = message.get('history_conversation', [])
        
        # 解析SQL生成过程和思考过程
        generation_steps = ""
        
        if history_conversation:
            try:
                # 找到system消息的位置
                system_indices = [i for i, x in enumerate(history_conversation) if x.get('role') == 'system']
                if system_indices:
                    idxs = system_indices[0]
                    his_llm = history_conversation[:idxs]  # SQL生成的历史对话
                    his_summrize = history_conversation[idxs+1:]  # agent思考过程
                    
                    # 提取assistant消息
                    llm_assistants = [msg.get('content', '') for msg in his_llm if msg.get('role') == 'assistant']
                    thought_user = [parse_query_result(msg.get('content', ''))  for msg in his_summrize if msg.get('role') == 'user']
                    thought_assistant = [prase_json_from_response(msg.get('content', '')) for msg in his_summrize if msg.get('role') == 'assistant']

                    for round_idx, sql_content in enumerate(llm_assistants):
                        res_sql = extract_xml_answer(sql_content)
                        res_sql = parse_sql_from_string(res_sql)
                        generation_steps += f"""
                        <div class="generation-step">
                            <h5>SQL生成 {round_idx + 1}:</h5>
                            <div class="code-block">
                                <pre><code>{res_sql}</code></pre>
                            </div>
                        </div>
                        """
                    
                        # 展示对应的思考过程
                        if round_idx < min(len(thought_user), len(thought_assistant)):
                            sql_query, action_thought = thought_user[round_idx], thought_assistant[round_idx]
                            generation_steps += f"""
                            <div class="thinking-step">
                                <strong>查询语句:</strong><p>{sql_query}</p>
                            </div>
                            """
                            generation_steps += f"""
                            <div class="thinking-query">
                                <strong>思考过程:</strong> {action_thought}
                            </div>
                            """

            except (IndexError, ValueError) as e:
                print(f"解析history_conversation时出错: {e}")
        
        result = f"""<div class="step-result">
        <h4>SQL生成与思考过程</h4>
        <div class="generation-thinking-section">
            {generation_steps if generation_steps else '<p>无生成记录</p>'}
        </div>
        
        <h4>查询结果</h4>
        <div class="answer-content">
            <pre>{answer}</pre>
        </div>
        </div>"""
        
        return result

    def _generate_table_from_agent_result(self, message: Dict) -> str:
        """生成表格数据"""
        sql = message.get('sql', '')
        answer = message.get('answer', '')
        
        if not sql or not answer:
            return "<div class='empty-state'>暂无查询结果</div>"
        
        # 解析SQL语句
        sql_parts = sql.split('SELECT')
        if len(sql_parts) < 2:
            return "<div class='empty-state'>无法解析SQL语句</div>"
        
        table_data = answer.split('\n')[1:]  # 提取表格数据部分
        
        if not table_data:
            return "<div class='empty-state'>查询结果为空</div>"
        
        # 解析表格数据
        data = []
        for row in table_data:
            cells = row.split('\t')
            if len(cells) > 1:  # 确保有数据
                data.append(cells)
        
        if not data:
            return "<div class='empty-state'>查询结果为空</div>"
        
        # 生成表格HTML
        headers = data[0]
        table_html = "<table class='result-table'>"
        table_html += "<tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr>"
        for row in data[1:]:
            table_html += "<tr>" + "".join([f"<td>{cell}</td>" for cell in row]) + "</tr>"
        table_html += "</table>"
        
        return table_html

def format_conversation_list(conversations: List[Dict], current_id: str = None) -> str:
    """格式化对话列表"""
    if not conversations:
        return "<div class='empty-state'>暂无历史对话</div>"
    
    html = ""
    for conv in conversations:
        active_class = "active" if conv["id"] == current_id else ""
        created_time = conv["created_at"].strftime("%m/%d %H:%M")
        
        html += f"""
        <div class="chat-item {active_class}" onclick="switchConversation('{conv["id"]}')">
            <div class="chat-title">{conv["title"]}</div>
            <div class="chat-time">{created_time}</div>
        </div>
        """
    
    return html

def format_chat_messages(messages: List[Dict]) -> str:
    """格式化对话消息"""
    if not messages:
        return "<div class='empty-state'>开始您的第一次查询吧！</div>"
    
    html = ""
    for msg in messages:
        if msg["role"] == "user":
            html += f"""
            <div class="chat-message user-message fade-in">
                <div class="message-bubble user-bubble">
                    {msg["content"]}
                    <div class="message-meta">第{msg["round"]}轮 · {msg["timestamp"]}</div>
                </div>
            </div>
            """
        else:
            html += f"""
            <div class="chat-message system-message fade-in">
                <div class="message-bubble system-bubble">
                    {msg["content"]}
                    <div class="message-meta">助手回复 · {msg["timestamp"]}</div>
                </div>
            </div>
            """
    
    return html

def format_enhanced_agent_process(agent_process: Dict) -> str:
    """增强的Agent思考过程格式化"""
    if not agent_process:
        return "<div class='empty-state'>等待查询开始...</div>"
    
    html = ""
    steps = [
        ("step1", "Step 1: Query解析", "🔍"),
        ("step2", "Step 2: Schema链接", "🔗"),
        ("step3", "Step 3: SQL生成", "⚡")
    ]
    
    for step_key, step_title, icon in steps:
        step_data = agent_process.get(step_key, {"status": "pending", "content": "", "logs": []})
        status = step_data["status"]
        content = step_data["content"]
        logs = step_data.get("logs", [])
        
        status_class = f"step-{status}"
        if status == "processing":
            status_icon = "<span class='loading-spinner'></span>"
        elif status == "success":
            status_icon = "✅"
        elif status == "error":
            status_icon = "❌"
        else:
            status_icon = "⏳"
        
        # 构建展开/折叠的内容
        detail_id = f"detail-{step_key}"
        
        html += f"""
        <div class="agent-step {status_class} fade-in">
            <div class="step-header" onclick="toggleStepDetail('{detail_id}')">
                <span>{icon} {step_title}</span>
                <span>{status_icon}</span>
            </div>
            <div class="step-content">
                <div class="step-summary">
                    {content}
                </div>
                <div id="{detail_id}" class="step-detail" style="display: none;">
                    <div class="log-section">
                        <h4>详细日志:</h4>
                        <div class="log-entries">
        """
        
        # 添加详细日志
        for log in logs:
            log_class = f"log-{log['type']}"
            html += f"""
                            <div class="log-entry {log_class}">
                                <span class="log-time">[{log['timestamp']}]</span>
                                <span class="log-content">{log['content']}</span>
                            </div>
            """
        
        html += """
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    
    return html

def format_sql_display(sql: str) -> str:
    """格式化SQL显示"""
    if not sql:
        return "<div class='empty-state'>暂无SQL语句</div>"
    
    return f"""
    <div class="sql-block">
        <div class="sql-header">Generated SQL</div>
        <pre>{sql}</pre>
    </div>
    <div class="tool-buttons">
        <button class="tool-btn" onclick="navigator.clipboard.writeText(`{sql}`)">📋 复制SQL</button>
        <button class="tool-btn">🎯 查看执行计划</button>
    </div>
    """

# 创建Gradio界面
def create_interface():
    chatbot = NL2SQLChatBot()
    
    # 使用分离的CSS样式
    css_styles = get_gradio_styles()
    
    with gr.Blocks(css=css_styles, title="NL2SQL智能助手", theme="soft") as demo:
        # 状态管理
        current_conversation_state = gr.State(None)
        agent_state = gr.State({})
        
        # 顶部导航栏
        gr.HTML("""
        <div class="header-bar">
            <div class="logo">🤖 NL2SQL智能助手</div>
            <div class="version">v1.0 | 金融数据查询助手</div>
        </div>
        """)
        
        # 主要布局
        with gr.Row():
            # 左侧：对话侧边栏 (20%)
            with gr.Column(scale=2):
                gr.HTML("<div class='sidebar-header'>💬 对话列表</div>")
                new_chat_btn = gr.Button("+ 新建对话", elem_classes=["new-chat-btn"])
                conversation_list = gr.HTML(
                    value="<div class='empty-state'>暂无历史对话</div>",
                    elem_classes=["sidebar"]
                )
            
            # 中间：主对话区 (50%)
            with gr.Column(scale=5):
                # 对话显示区域
                chat_display = gr.HTML(
                    value="<div class='empty-state'>开始您的第一次查询吧！</div>",
                    elem_classes=["chat-container"]
                )
                
                # 输入区域
                with gr.Group(elem_classes=["input-area"]):
                    with gr.Row():
                        user_input = gr.Textbox(
                            placeholder="请输入自然语言查询（如：截止到目前，湖南省共有多少家上市公司？）",
                            lines=2,
                            scale=4,
                            show_label=False,
                            elem_classes=["input-box"]
                        )
                        submit_btn = gr.Button("发送", scale=1, elem_classes=["send-button"])
            
            # 右侧：AI思考过程面板 (30%)
            with gr.Column(scale=3):
                gr.HTML("<div class='panel-title'>🔬 AI思考过程</div>")
                agent_display = gr.HTML(
                    value="<div class='empty-state'>等待查询开始...</div>",
                    elem_classes=["right-panel"]
                )

        # JavaScript代码
        gr.HTML("""
        <script>
        function switchConversation(conversationId) {
            const switchBtn = document.querySelector('#switch_conversation_btn');
            if (switchBtn) {
                switchBtn.click();
            }
        }

        function toggleStepDetail(detailId) {
            const detail = document.getElementById(detailId);
            if (detail) {
                detail.style.display = detail.style.display === 'none' ? 'block' : 'none';
            }
        }
        </script>
        """)

        # 隐藏的切换对话按钮
        switch_conversation_btn = gr.Button("Switch", visible=False, elem_id="switch_conversation_btn")

        # 事件处理 - 实时更新版本
        def handle_submit_realtime(user_input, current_conversation_id):
            if not user_input.strip():
                yield user_input, current_conversation_id, "", "", ""
                return
            
            # 使用实时更新的生成器
            for messages, agent_process_html in chatbot.process_query_with_realtime_updates(user_input):
                yield (
                    "",  # 清空输入框
                    chatbot.current_conversation_id,  # 更新当前对话ID
                    format_conversation_list(chatbot.get_conversation_list(), chatbot.current_conversation_id),  # 更新对话列表
                    format_chat_messages(messages),  # 更新聊天显示
                    agent_process_html  # 实时更新Agent过程
                )

        def create_new_chat():
            conversation_id = chatbot.create_new_conversation()
            return (
                conversation_id,  # 更新当前对话ID
                format_conversation_list(chatbot.get_conversation_list(), conversation_id),  # 更新对话列表
                "<div class='empty-state'>开始您的第一次查询吧！</div>",  # 清空聊天显示
                "<div class='empty-state'>等待查询开始...</div>"  # 清空Agent
            )

        # 绑定事件 - 使用实时更新
        submit_btn.click(
            handle_submit_realtime,
            inputs=[user_input, current_conversation_state],
            outputs=[
                user_input, current_conversation_state, conversation_list,
                chat_display, agent_display
            ]
        )
        
        user_input.submit(
            handle_submit_realtime,
            inputs=[user_input, current_conversation_state],
            outputs=[
                user_input, current_conversation_state, conversation_list,
                chat_display, agent_display
            ]
        )
        
        new_chat_btn.click(
            create_new_chat,
            outputs=[
                current_conversation_state, conversation_list, chat_display, agent_display
            ]
        )

    return demo

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("../output/gradio_temp", exist_ok=True)
    
    # 启动应用
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=8080,
        share=True,
        show_error=True,
        debug=True
    )

