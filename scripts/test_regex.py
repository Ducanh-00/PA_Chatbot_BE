import re

query = 'Cho tôi biết phạm vi áp dụng ISMS trong tài liệu xác định phạm vi'
doc_words = r'(?:file|tài liệu|tl|document|văn bản|quy trình|quy định)'
stop_words = (
    r'(?:,|\?|thì\b|là gì|là\b|có\b|nói\b|đề cập|quy định\b|mục đích|nội dung'
    r'|phạm vi|gồm|bao gồm|ghi\b|viết\b|trình bày|hướng dẫn|cho biết'
    r'|như thế nào|ra sao|ở đâu|bao nhiêu|khi nào|tại sao|để làm gì)'
)

file_ref_patterns = [
    rf'(?:nội dung|nd)\s+(?:của\s+)?{doc_words}\s+["\']?(.+?)["\']?\s*{stop_words}',
    rf'(?:nội dung|nd)\s+(?:của\s+)?{doc_words}\s+["\']?(.+?)["\']?\s*$',
    rf'(?:trong\s+)?{doc_words}\s+["\']?(.+?)["\']?\s*{stop_words}',
    rf'(?:trong\s+)?{doc_words}\s+["\']?(.+?)["\']?\s*(?:\?|$)',
]

for p in file_ref_patterns:
    m = re.search(p, query, re.IGNORECASE)
    if m:
        print("MATCHED PATTERN:", p)
        print("GROUP 1:", m.group(1))
        break

code_match = re.findall(r'((?:QT|QD|QĐ|BM|ISMS|SMS)[.\s-][A-Za-z]+[.\s-]?\d*)', query, re.IGNORECASE)
print("CODE MATCH:", code_match)
