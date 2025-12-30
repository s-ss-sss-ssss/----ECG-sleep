# readme

文件目录结构：

```
.
├── ISRUC
│   ├── processed
│   │   ├── 001_ecg_epochs.npy
│   │   ├── 001_fs.npy
...
│   │   ├── 100_fs.npy
│   │   └── 100_labels.npy
│   ├── subjects
│   │   ├── 001
│   │   │   ├── 001.edf
│   │   │   ├── 001_1.txt
│   │   │   ├── 001_1.xlsx
│   │   │   ├── 001_2.txt
│   │   │   └── 001_2.xlsx
... # 若使用缓存训练，subject文件夹留空即可
├── data.py
├── deep_learning.py
├── deep_learning_model.py
├── deep_learning_preprocess.py
├── isruc_extract.sh
├── rename.sh
├── rpeaks.py
└── training_log.txt
```

