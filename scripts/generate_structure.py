"""
项目目录结构自动生成脚本

用法：
  python scripts/generate_structure.py            # 打印到控制台
  python scripts/generate_structure.py --write     # 写入 运行命令.md 的项目结构章节

扫描项目目录、读取 docs/file_descriptions.json 的描述，生成格式化的目录树。
"""
import os
import json
import argparse
import fnmatch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESC_FILE = os.path.join(PROJECT_ROOT, "docs", "file_descriptions.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "运行命令.md")

# 排除的目录和文件模式（同 .gitignore 风格）
EXCLUDE_DIRS = {
    "__pycache__", ".git", ".github", ".omo", ".opencode",
    "node_modules", "__pycache__", ".git", "output",
}
EXCLUDE_PATTERNS = {
    "*.pyc", "*.log", "streamlit.*", ".env",
    "*.db-shm", "*.db-wal",
}


def load_descriptions() -> dict:
    """加载文件描述清单"""
    if os.path.exists(DESC_FILE):
        with open(DESC_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 去掉 _meta 键
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return {}


def should_exclude(name: str) -> bool:
    """检查是否应该排除"""
    if name in EXCLUDE_DIRS:
        return True
    for pat in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def build_tree(start_path: str, prefix: str = "") -> list[str]:
    """递归扫描目录，返回格式化的目录树行"""
    lines = []
    entries = sorted(
        e for e in os.listdir(start_path)
        if not e.startswith(".") and not should_exclude(e)
    )
    # 目录排在前面
    dirs = [e for e in entries if os.path.isdir(os.path.join(start_path, e))]
    files = [e for e in entries if not os.path.isdir(os.path.join(start_path, e))]
    # 按特定顺序排序：__init__.py 放前，其余按字母
    for lst in [dirs, files]:
        lst.sort(key=lambda x: (x != "__init__.py", x))

    all_entries = dirs + files
    for i, name in enumerate(all_entries):
        is_last = i == len(all_entries) - 1
        connector = "└── " if is_last else "├── "
        sub_prefix = "    " if is_last else "│   "
        full_path = os.path.join(start_path, name)
        rel_path = os.path.relpath(full_path, PROJECT_ROOT).replace("\\", "/")

        desc = descriptions.get(rel_path) or descriptions.get(rel_path + "/") or ""
        desc_str = f"  — {desc}" if desc else ""

        if os.path.isdir(full_path):
            lines.append(f"{prefix}{connector}{name}/{desc_str}")
            lines.extend(build_tree(full_path, prefix + sub_prefix))
        else:
            lines.append(f"{prefix}{connector}{name}{desc_str}")

    return lines


def generate_markdown() -> str:
    """生成完整的目录结构 Markdown"""
    global descriptions
    descriptions = load_descriptions()

    tree_lines = build_tree(PROJECT_ROOT)

    md = f"""# ETF 投资策略 — 项目目录结构

> 自动生成于 {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}
> 源描述文件: `docs/file_descriptions.json`
> 生成脚本: `scripts/generate_structure.py`

```
ETF投资策略/
{chr(10).join(tree_lines)}
```

### 维护说明

- **新增文件** → 在 `docs/file_descriptions.json` 中添加路径和描述
- **移动文件** → 路径变了需要同步更新 JSON 中的 key
- **重新生成** → `python scripts/generate_structure.py --write`
- 每次运行 `morning_update.py` 或 `etf-agent.py` 时自动更新
"""
    return md


def update_run_command_md(tree_md: str):
    """更新 运行命令.md 中的项目结构章节"""
    if not os.path.exists(OUTPUT_FILE):
        print(f"⚠️  未找到 {OUTPUT_FILE}，跳过写入")
        return

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # 找到项目结构章节的起始和结束标记
    marker_start = "## 📁 项目结构"
    marker_end = "\n---"

    start_idx = content.find(marker_start)
    if start_idx == -1:
        print(f"⚠️  未在 {OUTPUT_FILE} 中找到「{marker_start}」标记，跳过写入")
        return

    # 提取结构章节之前的内容
    before = content[:start_idx]

    # 只取树形图部分（去掉文档头，因为运行命令.md已有标题）
    tree_lines = tree_md.split("\n")
    # 跳过文档标题和自动生成说明，只保留 ``` 块内的内容
    in_code = False
    code_lines = []
    for line in tree_lines:
        if line.strip() == "```":
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)

    structure_section = (
        "\n\n" + marker_start + "\n\n```\n" +
        "\n".join(code_lines) +
        "\n```\n"
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(before + structure_section)

    print(f"✅ 已更新 {OUTPUT_FILE}")


def main():
    parser = argparse.ArgumentParser(description="生成项目目录结构文档")
    parser.add_argument("--write", action="store_true", help="写入 运行命令.md")
    args = parser.parse_args()

    md = generate_markdown()

    if args.write:
        update_run_command_md(md)
    else:
        print(md)


if __name__ == "__main__":
    main()
