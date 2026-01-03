# int_audit.py
# 目的:
#  - 対象 .py ファイル内の int(...) 呼び出しを機械的に列挙
#  - ざっくり「置換候補 / 要注意 / 不明」に分類
#  - 行番号・該当行・引数テキストを出力（CSVも出せる）
#
# 使い方:
#   python int_audit.py path/to/your_file.py
#   python int_audit.py path/to/your_file.py --csv out.csv
#
# 注意:
#  - これは「ASTベース」で int( ... ) だけを拾います（文字列中やコメントは拾いません）
#  - safe_int に置換するか最終判断は別途（ここは棚卸し用）

import ast
import argparse
import csv
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class IntCall:
    lineno: int
    col: int
    arg_src: str
    line_text: str
    category: str
    reason: str


def _get_source_segment(source: str, node: ast.AST) -> str:
    seg = ast.get_source_segment(source, node)
    if seg is None:
        return ""
    return seg.strip()


def _get_line(source_lines: List[str], lineno: int) -> str:
    if 1 <= lineno <= len(source_lines):
        return source_lines[lineno - 1].rstrip("\n")
    return ""


def _categorize(arg_src: str, full_line: str) -> Tuple[str, str]:
    """
    雑に分類（ルールは必要に応じて増やしてOK）:
      - replace_candidate: スライス/文字列由来が濃厚で壊れやすい -> safe_int候補
      - caution: 意味が壊れやすい（race_time分解、計算、index等） -> むやみに置換NG
      - unknown: 判断材料不足
    """
    s = arg_src.replace(" ", "")

    # 1) 明らかに「固定幅テキストのスライス」っぽい
    #    例: payoff_result[0][25:29], race_head[2:4], line[62:65], racer_result[2:4]
    if "[" in s and ":" in s and "]" in s:
        return ("replace_candidate", "slice-based parse (fixed-width text)")

    # 2) 文字列掃除してから int(...) してるやつ
    #    例: int(x.replace(...)), int(x.strip()), int(x.split(...)[...])
    if ".replace(" in s or ".strip(" in s:
        return ("replace_candidate", "string-cleaning before int()")

    # 3) splitで分解して数値化（race_time系に多い）: 例 int(race_time[0])
    if "split(" in s or "race_time[" in s:
        return ("caution", "time/token decomposition; naive safe_int can create false 'valid' numbers")

    # 4) すでに数字っぽいものを int にしているだけ（0/1など）
    #    例: int(racer_result[38]) は文字1桁由来で壊れやすいが、slice判定に入らないケース
    #        -> ここはunknown扱いにして手確認へ
    if s.isdigit():
        return ("unknown", "literal int() (usually safe; no replacement needed)")

    # 5) 計算式 / 乗算が入る（race_time変換など） -> 注意
    if "*" in s or "+" in s or "-" in s or "/" in s:
        return ("caution", "numeric expression; replacing can change semantics")

    # 6) それ以外
    #    例: int(frame) とか int(something) がどの程度汚れるか不明
    return ("unknown", "needs manual review")


class IntCallVisitor(ast.NodeVisitor):
    def __init__(self, source: str):
        self.source = source
        self.lines = source.splitlines(True)
        self.found: List[IntCall] = []

    def visit_Call(self, node: ast.Call):
        # int(...)
        if isinstance(node.func, ast.Name) and node.func.id == "int":
            lineno = getattr(node, "lineno", 0)
            col = getattr(node, "col_offset", 0)

            arg_src = ""
            if node.args:
                arg_src = _get_source_segment(self.source, node.args[0])
            full_line = _get_line(self.lines, lineno)

            category, reason = _categorize(arg_src, full_line)

            self.found.append(
                IntCall(
                    lineno=lineno,
                    col=col,
                    arg_src=arg_src,
                    line_text=full_line.strip(),
                    category=category,
                    reason=reason,
                )
            )
        self.generic_visit(node)


def audit_int_calls(py_path: str) -> List[IntCall]:
    with open(py_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=py_path)
    v = IntCallVisitor(source)
    v.visit(tree)
    return sorted(v.found, key=lambda x: (x.lineno, x.col))


def print_report(items: List[IntCall]):
    from collections import Counter
    c = Counter([x.category for x in items])

    print("=== int(...) audit report ===")
    print(f"total: {len(items)}")
    print("by category:", dict(c))
    print()

    for x in items:
        print(f"L{x.lineno}:{x.col}  [{x.category}]  {x.reason}")
        print(f"  arg:  {x.arg_src}")
        print(f"  line: {x.line_text}")
        print()


def write_csv(items: List[IntCall], out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lineno", "col", "category", "reason", "arg_src", "line_text"])
        for x in items:
            w.writerow([x.lineno, x.col, x.category, x.reason, x.arg_src, x.line_text])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="target python file path")
    ap.add_argument("--csv", default=None, help="optional: write csv to this path")
    args = ap.parse_args()

    items = audit_int_calls(args.path)
    print_report(items)
    if args.csv:
        write_csv(items, args.csv)
        print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
