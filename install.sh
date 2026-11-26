#!/bin/sh
# 把 interview-coach skill 以软链接装到各 agent 的用户级 skill 目录。
# 软链接指向本仓库，所以 git pull 更新知识库后 skill 自动生效。
#
# 用法：./install.sh [claude|codex|all]   默认 all
# 卸载：删除对应目录下的 interview-coach 软链接即可。
set -eu

repo=$(cd "$(dirname "$0")" && pwd)
src="$repo/skills/interview-coach"
target=${1:-all}

link() {
  dir=$1
  mkdir -p "$dir"
  dest="$dir/interview-coach"
  if [ -L "$dest" ]; then
    rm "$dest"
  elif [ -e "$dest" ]; then
    echo "跳过 $dest：已存在且不是软链接，请手动处理。" >&2
    return
  fi
  ln -s "$src" "$dest"
  echo "已安装：$dest -> $src"
}

case "$target" in
  claude) link "$HOME/.claude/skills" ;;
  codex)  link "$HOME/.agents/skills" ;;
  all)    link "$HOME/.claude/skills"; link "$HOME/.agents/skills" ;;
  *) echo "用法：$0 [claude|codex|all]" >&2; exit 2 ;;
esac

python3 "$src/scripts/coach.py" plan --list >/dev/null && echo "知识库可读，安装完成。数据目录：${AIKB_HOME:-$HOME/.agent-interview-kb}"
