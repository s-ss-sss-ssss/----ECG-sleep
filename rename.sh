#!/bin/bash
# 递归查找当前目录及子目录下的所有文件和文件夹中的*.rec文件，并将其重命名为*.edf文件
echo "=== ISRUC 数据集重命名脚本 ==="
echo "当前目录为" "$(pwd)"
echo "press Ctrl+C to cancel, or any key to continue..."
read -n 1 -s
find . -type f -name "*.rec" | while read file; do
    newfile="${file%.rec}.edf"
    mv "$file" "$newfile"
    echo "Renamed: $file -> $newfile"
done
echo "✅ 全部重命名完成！"