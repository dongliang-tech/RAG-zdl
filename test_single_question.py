import os
import sys
sys.path.insert(0, '/Users/wan/Desktop/测试用/AI大模型应用第17期/23-项目实战：企业知识库/RAG-cy')

os.environ['DASHSCOPE_API_KEY'] = 'sk-ws-H.RYEREXE.gSAX.MEUCIQC4H_PbsQEGngfAnwQokhVwmawCE2YY9UCzQPVLiBSVzAIgYGhRaqfKE6mZoVuL3A80anXvo9spIpsZ4TxetPDLPTs'

from src.questions_processing import QuestionsProcessor
from pathlib import Path

root_path = Path('/Users/wan/Desktop/测试用/AI大模型应用第17期/23-项目实战：企业知识库/RAG-cy/data/stock_data')
questions_file_path = root_path / 'questions.json'
vector_db_dir = root_path / 'databases' / 'vector_dbs'
chunked_reports_dir = root_path / 'databases' / 'chunked_reports'

processor = QuestionsProcessor(
    questions_file_path=questions_file_path,
    vector_db_dir=vector_db_dir,
    documents_dir=chunked_reports_dir,
    api_provider="dashscope",
    answering_model="deepseek-v4-pro"  # 尝试使用 deepseek-v4-pro
)

question = "中芯国际在晶圆制造行业中的地位如何？"
print(f"问题: {question}")
print("正在处理...")

result = processor.get_answer_for_company(company_name="中芯国际", question=question, schema="string")
print(f"\n结果: {result}")