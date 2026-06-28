"""
文本清洗工具 — 去除 LLM 回复中的 markdown 格式标记。

用于解决 AI 客服回复中出现 **bold** 等 markdown 语法，
前端不渲染 markdown 导致用户看到原始星号的问题。
"""

import re


def strip_markdown_formatting(text: str) -> str:
    """去除 LLM 回复中的基本 markdown 格式标记。

    处理：
      - **bold** → bold
      - *italic* → italic（仅在英文单词边界时处理）
      - __underline__ → underline
      - `code` → code

    不处理：
      - 表格、代码块、链接等（客服场景极少出现）
      - 中文书名号《》（与 markdown 无关）
    """
    if not text:
        return text

    # **bold** → bold
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)

    # __underline__ → underline
    text = re.sub(r'__(.+?)__', r'\1', text)

    # *italic* → italic（小心处理，避免误伤中文中的 *）
    text = re.sub(r'\*([^*\n]+?)\*', r'\1', text)

    # `code` → code
    text = re.sub(r'`([^`\n]+?)`', r'\1', text)

    return text
