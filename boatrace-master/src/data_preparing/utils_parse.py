import re

def safe_slice(s, start, end):
    """
    文字列を安全に切り出す
    - 短くても落ちない
    - Noneでも落ちない
    """
    if not s:
        return ""
    if start >= len(s):
        return ""
    return s[start:end]


def safe_char(s, index):
    """
    1文字を安全に取る
    """
    if not s:
        return None
    if index < 0 or index >= len(s):
        return None
    return s[index]


def safe_int(text, default=None):
    """
    数字を安全に int にする
    - 空白OK
    - '0  人' OK
    - 数字がなければ default
    """
    if text is None:
        return default

    text = str(text)
    m = re.search(r"\d+", text)
    if not m:
        return default

    return int(m.group())
