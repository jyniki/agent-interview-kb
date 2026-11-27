#!/bin/sh
# 把 interview-coach skill 以软链接装到各 agent 的用户级 skill 目录。
# 软链接指向本仓库，所以 git pull 更新知识库后 skill 自动生效。
#
# 用法：./install.sh [claude|codex|dsh|all]   默认 all
#   claude   Claude Code        ~/.claude/skills
#   codex    Codex CLI          ~/.codex/skills、~/.agents/skills（两个都装，不同版本读不同位置）
#   dsh      dsh                ~/.dsh/skills
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

install_claude() { link "$HOME/.claude/skills"; }
install_codex()  { link "$HOME/.codex/skills"; link "$HOME/.agents/skills"; }
install_dsh()    { link "$HOME/.dsh/skills"; }

case "$target" in
  claude) install_claude ;;
  codex)  install_codex ;;
  dsh)    install_dsh ;;
  all)    install_claude; install_codex; install_dsh ;;
  *) echo "用法：$0 [claude|codex|dsh|all]" >&2; exit 2 ;;
esac

python3 "$src/scripts/coach.py" plan --list >/dev/null && echo "知识库可读，安装完成。数据目录：${AIKB_HOME:-$HOME/.agent-interview-kb}"
