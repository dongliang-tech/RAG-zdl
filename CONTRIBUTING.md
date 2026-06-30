# 贡献指南

首先，感谢你考虑为 Enterprise-RAG 项目做出贡献！

## 如何贡献

### 报告 Bug

如果你发现了 Bug，请创建一个 GitHub Issue，并提供：

1. 清晰的标题和描述
2. 复现步骤
3. 期望的行为和实际行为
4. 环境信息（Python 版本、依赖版本等）
5. 如有可能，附上错误日志或截图

### 建议新功能

如果你有新功能的建议，请先创建一个 GitHub Issue 进行讨论，包括：

1. 功能描述
2. 使用场景
3. 预期行为
4. 是否有替代方案

### 提交代码

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

### 代码规范

- 遵循 PEP 8 Python 代码风格规范
- 代码注释使用中文或英文均可，保持与所在文件一致的语言
- 新增功能请尽量添加相应的测试
- 确保所有测试通过后再提交 PR

## 开发环境设置

1. 克隆仓库并安装依赖：

```bash
git clone https://github.com/dongliang-tech/Enterprise-RAG.git
cd Enterprise-RAG
pip install -r requirements.txt
```

2. 配置环境变量：

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥
```

3. 运行项目：

```bash
# 运行 Streamlit Web UI
python run.sh

# 或使用命令行
python main.py --help
```

## 代码审查

所有提交到主分支的代码都需要经过代码审查。PR 标题请遵循以下格式：

```
[模块名]: 简短描述
```

例如：
- `[pipeline] 优化 PDF 解析流程`
- `[retrieval] 修复混合检索中的边界条件`
- `[docs] 更新 README 中的配置说明`

## 问题解答

如果你在贡献过程中有任何疑问，可以：

1. 查看项目的 README.md 文档
2. 在 GitHub Issue 中提问
3. 查看现有的 Issue 和 PR

## 行为准则

我们期望所有贡献者遵守以下准则：

- 使用友好和包容的语言
- 尊重不同的观点和经验
- 建设性地接受批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

## 致谢

感谢所有为本项目做出贡献的人！
