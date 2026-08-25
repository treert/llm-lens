"""扫描并修复 markdown 中行内公式开头 `$` 前缺少空格的问题。

GitHub 渲染限制:行内公式开头的 `$` 若紧跟文字(中文/英文/数字)或全角标点
(如 `（$`),公式不会被渲染;在 `$` 前补一个空格即可。
本工具扫描 markdown 文件找出这类位置,默认只报告(dry-run),`--apply` 时原地修复。

规则细节:
- `$` 前一字符是空白、行首或 ASCII 标点(如 `(`、`*`)时不处理:GitHub 能正常
  渲染,且在 `**$x$**` 这类粗体中插空格反而会破坏语法
- 跳过围栏代码块(``` / ~~~)、行内代码、`$$` 块级公式、`\\$` 转义
- 不处理 4 空格缩进代码块(本仓库文档均用围栏代码块)

用法:
    python tools/fix_md_math_spacing.py            # 扫描全仓库,只报告
    python tools/fix_md_math_spacing.py --apply    # 原地修复
    python tools/fix_md_math_spacing.py docs README.md --apply   # 只处理指定文件/目录

退出码:0 = 干净,1 = 发现待修复位置(可用于 pre-commit / CI)。
"""

from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {".git", ".codebuddy", ".venv", "venv", "node_modules", "output", "tmp"}

# `$` 前是这些字符时无需补空格:空白,以及 ASCII 标点(GitHub 可正常渲染,
# 且能避免破坏 `**$x$**` 这类粗体语法)
_NO_FIX_BEFORE = frozenset(string.whitespace + string.punctuation)


def find_fix_positions(lines: list[str]) -> list[tuple[int, int]]:
    """扫描整个文件,返回需补空格的位置 [(行号, 列号)](均从 0 开始)。"""
    positions: list[tuple[int, int]] = []
    in_fence = False
    fence_char, fence_len = "", 0
    in_display = False  # `$$` 块级公式,可跨行

    for ln, line in enumerate(lines):
        stripped = line.lstrip()
        if in_fence:
            if stripped.startswith(fence_char * fence_len):
                in_fence = False
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = True
            fence_char = stripped[0]
            fence_len = len(stripped) - len(stripped.lstrip(fence_char))
            continue

        in_math = False  # 行内公式不跨行
        i, n = 0, len(line)
        while i < n:
            ch = line[i]
            if in_display:
                if line.startswith("$$", i):
                    in_display = False
                    i += 2
                else:
                    i += 1
                continue
            if ch == "`":  # 行内代码整段跳过(反引号个数需配对)
                run = 1
                while i + run < n and line[i + run] == "`":
                    run += 1
                closing = line.find("`" * run, i + run)
                i = n if closing == -1 else closing + run
                continue
            if ch == "\\" and i + 1 < n and line[i + 1] in "\\$":  # 转义字符
                i += 2
                continue
            if ch == "$":
                if line.startswith("$$", i):  # 块级公式
                    closing = line.find("$$", i + 2)
                    if closing == -1:
                        in_display = True
                        i = n
                    else:
                        i = closing + 2
                    continue
                if in_math:
                    in_math = False
                else:
                    in_math = True
                    if i > 0 and line[i - 1] not in _NO_FIX_BEFORE:
                        positions.append((ln, i))
                i += 1
                continue
            i += 1
    return positions


def _snippet(body: str, col: int, half: int = 25) -> str:
    """截取 `$` 附近的上下文,用于报告。"""
    start, end = max(0, col - half), min(len(body), col + half)
    return ("…" if start else "") + body[start:end] + ("…" if end < len(body) else "")


def process_file(path: Path, apply: bool) -> list[tuple[int, int, str]]:
    """处理单个文件,返回 [(行号, 列号, 上下文)];apply 为 True 时原地修复。"""
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            raw_lines = fh.read().splitlines(keepends=True)
    except UnicodeDecodeError:
        print(f"警告:{path} 不是有效 UTF-8,已跳过", file=sys.stderr)
        return []
    bodies = [line.rstrip("\r\n") for line in raw_lines]
    positions = find_fix_positions(bodies)

    if apply and positions:
        by_line: dict[int, list[int]] = {}
        for ln, col in positions:
            by_line.setdefault(ln, []).append(col)
        fixed = list(raw_lines)
        for ln, cols in by_line.items():
            body = bodies[ln]
            for col in sorted(cols, reverse=True):  # 从后往前插,列号不失效
                body = body[:col] + " " + body[col:]
            fixed[ln] = body + raw_lines[ln][len(bodies[ln]):]  # 保留原换行符
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("".join(fixed))

    return [(ln, col, _snippet(bodies[ln], col)) for ln, col in positions]


def iter_markdown_files(paths: list[Path]) -> list[Path]:
    """展开目录为 markdown 文件列表,并排除 EXCLUDE_DIRS / *.egg-info。"""
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(p.rglob("*.md"))
        elif p.is_file():
            files.append(p)
    return sorted(
        {f for f in files
         if not any(part in EXCLUDE_DIRS or part.endswith(".egg-info") for part in f.parts)}
    )


def _rel(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="修复 markdown 行内公式开头 `$` 前缺空格导致 GitHub 不渲染的问题")
    parser.add_argument("paths", nargs="*", type=Path,
                        help="要扫描的文件/目录,默认整个仓库(排除 .venv、node_modules 等)")
    parser.add_argument("--apply", action="store_true", help="原地修复;默认只报告")
    args = parser.parse_args()

    files = iter_markdown_files(args.paths or [REPO_ROOT])
    if not files:
        print("未找到 markdown 文件")
        return 0

    total, dirty = 0, 0
    for f in files:
        hits = process_file(f, apply=args.apply)
        if not hits:
            continue
        dirty += 1
        total += len(hits)
        for ln, col, ctx in hits:
            print(f"{_rel(f)}:{ln + 1}:{col + 1}: {ctx}")

    if total == 0:
        print(f"扫描 {len(files)} 个文件,无需修复。")
        return 0
    print(f"\n{'已修复' if args.apply else '待修复'} {total} 处,"
          f"涉及 {dirty}/{len(files)} 个文件。")
    if not args.apply:
        print("确认无误后加 --apply 写回修改。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
