import streamlit as st
from pathlib import Path
from src.pipeline import Pipeline, max_config
from src.questions_processing import QuestionsProcessor
import json
import traceback

root_path = Path("data/stock_data")
pipeline = Pipeline(root_path, run_config=max_config)

st.set_page_config(page_title="RAG Challenge 2", layout="wide")

theme = st.get_option('theme.base')
if theme == 'dark':
    bg_color = '#1e1e1e'
    text_color = '#e8e8e8'
    border_color = '#28a745'
    status_text_color = '#4ade80'
    generating_bg = '#1a1a2e'
    generating_text = '#a0a0a0'
else:
    bg_color = '#f6f8fa'
    text_color = '#333333'
    border_color = '#28a745'
    status_text_color = '#28a745'
    generating_bg = '#f0f8ff'
    generating_text = '#888888'

st.markdown("""
<div style='background: linear-gradient(90deg, #7b2ff2 0%, #f357a8 100%); padding: 20px 0; border-radius: 12px; text-align: center;'>
    <h2 style='color: white; margin: 0;'>🚀 企业知识库 RAG 系统</h2>
    <div style='color: #fff; font-size: 16px;'>多公司年报问答 | FAISS向量检索 + BM25关键词检索 + Jina重排 | 表格序列化 + 父文档检索 | DashScope/Qwen/OpenAI多模型支持 | 缓存加速 | 流式输出</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("查询设置")
    company_name = st.text_input("公司名称", "中芯国际", help="请输入要查询的公司名称，需与年报中的公司名一致")
    user_question = st.text_area("输入问题", "请简要总结公司2022年主营业务的主要内容。", height=80)
    submit_btn = st.button("生成答案", use_container_width=True)

st.markdown("<h3 style='margin-top: 24px;'>检索结果</h3>", unsafe_allow_html=True)

if submit_btn and user_question.strip():
    status_placeholder = st.empty()
    answer_placeholder = st.empty()
    step_by_step_placeholder = st.empty()
    reasoning_summary_placeholder = st.empty()
    relevant_pages_placeholder = st.empty()
    
    status_placeholder.info("🔄 正在检索相关信息，请稍候...")
    
    try:
        answer_dict = {
            "step_by_step_analysis": "",
            "reasoning_summary": "",
            "relevant_pages": [],
            "final_answer": ""
        }
        
        for event in pipeline.answer_single_question_streaming(user_question, kind="string", company_name=company_name.strip()):
            event_type = event.get("event")
            
            if event_type == "retrieval_done":
                status_placeholder.info("✅ 检索完成，正在生成答案...")
            elif event_type == "streaming_token":
                token = event.get("token", "")
                answer_dict["final_answer"] += token
                answer_placeholder.markdown(f"""
<div style='background:{generating_bg};padding:16px;border-radius:8px;font-size:15px;border-left:4px solid #7b2ff2;color:{text_color};'>
<p style='color:{generating_text};font-size:13px;margin-bottom:8px;'>⏳ 正在生成答案中...</p>
{answer_dict["final_answer"]}
</div>
""", unsafe_allow_html=True)
            elif event_type == "step_by_step":
                answer_dict["step_by_step_analysis"] = event.get("content", "")
                step_by_step_placeholder.markdown(f"**分步推理：**\n\n{event.get('content', '')}")
            elif event_type == "reasoning_summary":
                answer_dict["reasoning_summary"] = event.get("content", "")
                reasoning_summary_placeholder.success(f"**推理摘要：**\n\n{event.get('content', '')}")
            elif event_type == "relevant_pages":
                pages = event.get("content", [])
                answer_dict["relevant_pages"] = pages
                if isinstance(pages, list) and len(pages) > 0:
                    page_numbers = []
                    for p in pages:
                        if isinstance(p, (list, tuple)):
                            page_numbers.append(str(p[0]))
                        elif isinstance(p, dict):
                            page_numbers.append(str(p.get('page', '')))
                        else:
                            page_numbers.append(str(p))
                    pages_text = ", ".join(page_numbers)
                    relevant_pages_placeholder.markdown(f"**📄 相关页面：** {pages_text}")
                else:
                    relevant_pages_placeholder.markdown("**📄 相关页面：** 未找到相关页面")
            elif event_type == "final_answer":
                answer_dict["final_answer"] = event.get("content", "")
                status_placeholder.success("✅ 答案生成完成")
                answer_placeholder.markdown(f"""
<div style='background:{bg_color};padding:16px;border-radius:8px;font-size:17px;border-left:4px solid {border_color};color:{text_color};'>
<p style='color:{status_text_color};font-size:13px;margin-bottom:8px;'>✅ 答案生成完成</p>
{event.get('content', '')}
</div>
""", unsafe_allow_html=True)
            elif event_type == "complete":
                pass
            elif event_type == "error":
                status_placeholder.empty()
                st.error(f"生成答案时出错: {event.get('error', '')}")
                st.stop()
                
    except Exception as e:
        status_placeholder.empty()
        st.error(f"生成答案时出错: {e}")
        st.error("详细错误信息:")
        st.code(traceback.format_exc())
else:
    st.info("请在左侧输入问题并点击【生成答案】") 