#!/bin/bash
echo "=== ISRUC 数据集解压脚本 ==="
echo "当前目录为" "$(pwd)"
echo "press Ctrl+C to cancel, or any key to continue..."
read -n 1 -s

RAR_DIR="/mnt/d/isruc"  
OUT_DIR="./ISRUC/subjects"

mkdir -p "$OUT_DIR"

# 检查 unrar
if ! command -v unrar &> /dev/null
then
    echo "unrar 未安装，请执行: sudo apt install unrar"
    exit 1
fi

for i in $(seq 1 100)
do
    ID=$(printf "%03d" $i)     # 001, 002, ...
    RAR_FILE="${RAR_DIR}/${i}.rar"

    echo "====== 解压 Subject ${ID} ======"

    if [ ! -f "$RAR_FILE" ]; then
        echo "⚠ 找不到 ${RAR_FILE}，跳过"
        continue
    fi

    SUBJECT_DIR="${OUT_DIR}/${ID}"
    mkdir -p "$SUBJECT_DIR"

    # 临时解压目录
    TEMP_DIR=$(mktemp -d)

    # 解压
    unrar x "$RAR_FILE" "$TEMP_DIR" >/dev/null

    # 找到压缩包内部的目录（如 1/）
    INNER_DIR=$(find "$TEMP_DIR" -maxdepth 1 -type d | tail -n +2)

    # 重命名并移动文件
    for f in "$INNER_DIR"/*; do
        fname=$(basename "$f")
        newname=$(echo "$fname" | sed "s/^[0-9]\+/$ID/")
        mv "$f" "${SUBJECT_DIR}/${newname}"
    done

    rm -rf "$TEMP_DIR"
done

echo "✅ 全部解压完成，数据位于 ISRUC/subjects/"
