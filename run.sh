#!/bin/bash
# RAG 应用启动脚本（支持热重载）
# 修改代码后会自动刷新页面，无需手动重启

echo "🚀 启动 RAG 应用（热重载模式）..."
echo "💡 修改代码后会自动刷新页面"
echo ""

streamlit run app_streamlit.py --browser.gatherUsageStats false
