from markitdown import MarkItDown

markitdown = MarkItDown()
result = markitdown.convert("2408.17142v1.pdf")

# 結果をMarkdownファイルとして保存します
with open('output.md', 'w', encoding='utf-8') as f:
    f.write(result.text_content)