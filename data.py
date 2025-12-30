import numpy as np
from pathlib import Path
import pyedflib
import time

class ISRUC_ECG_Reader:
    """
    ISRUC ECG Reader with:
    - automatic EDF -> numpy preprocessing
    - 30s epoch segmentation
    """

    def __init__(
        self,
        root_dir,
        processed_dir=None,
        ecg_channel=9,
        epoch_sec=30
    ):
        """
        Parameters
        ----------
        root_dir : str or Path
            ISRUC/subjects
        processed_dir : str or Path or None
            若为 None，则默认为 root_dir/../processed
        ecg_channel : int
            ECG 通道索引（0-based）
        epoch_sec : int
            每个 epoch 的秒数（默认 30s）
        """
        self.root_dir = Path(root_dir)
        self.ecg_channel = ecg_channel
        self.epoch_sec = epoch_sec

        self.processed_dir = (
            Path(processed_dir)
            if processed_dir is not None
            else self.root_dir.parent / "processed"
        )
        self.processed_dir.mkdir(exist_ok=True)

        # 首选使用真实的 subjects 目录；如果不存在或为空，则从 processed 文件中推断 subject id
        subjects_dirs = []
        if self.root_dir.exists():
            subjects_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]

        if subjects_dirs:
            self.subjects = sorted(subjects_dirs)
        else:
            # 从 processed 目录中寻找形如 <sid>_ecg_epochs.npy 的文件来推断 sid
            sids = set()
            if self.processed_dir.exists():
                for p in self.processed_dir.iterdir():
                    name = p.name
                    if name.endswith("_ecg_epochs.npy"):
                        sid = name[: -len("_ecg_epochs.npy")]
                        if sid:
                            sids.add(sid)
            self.subjects = sorted(Path(self.processed_dir / sid) for sid in sids)

    def __len__(self):
        return len(self.subjects)

    # ------------------------------------------------------------------
    # 核心工具函数
    # ------------------------------------------------------------------
    def _segment_ecg(self, ecg, labels, fs):
        """
        ECG -> 30s epoch segmentation & label alignment
        """
        samples_per_epoch = int(fs * self.epoch_sec)
        num_epochs = min(
            len(ecg) // samples_per_epoch,
            len(labels)
        )

        ecg = ecg[: num_epochs * samples_per_epoch]
        ecg_epochs = ecg.reshape(num_epochs, samples_per_epoch)
        labels = labels[:num_epochs]

        return ecg_epochs, labels

    # ------------------------------------------------------------------
    # EDF 读取（慢）
    # ------------------------------------------------------------------
    def _load_from_edf(self, subject_dir):
        sid = subject_dir.name

        edf_path = subject_dir / f"{sid}.edf"
        label_path = subject_dir / f"{sid}_1.txt"

        if not edf_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Missing EDF or label for {sid}")

        with pyedflib.EdfReader(str(edf_path)) as f:
            ecg = f.readSignal(self.ecg_channel)
            fs = f.getSampleFrequency(self.ecg_channel)

        labels = np.loadtxt(label_path, dtype=int, ndmin=1)

        ecg_epochs, labels = self._segment_ecg(ecg, labels, fs)

        return ecg_epochs, labels, fs

    # ------------------------------------------------------------------
    # numpy IO（快）
    # ------------------------------------------------------------------
    def _processed_paths(self, sid):
        return {
            "ecg": self.processed_dir / f"{sid}_ecg_epochs.npy",
            "labels": self.processed_dir / f"{sid}_labels.npy",
            "fs": self.processed_dir / f"{sid}_fs.npy",
        }

    def _is_processed(self, sid):
        paths = self._processed_paths(sid)
        return all(p.exists() for p in paths.values())

    def _save_processed(self, sid, ecg_epochs, labels, fs):
        paths = self._processed_paths(sid)
        np.save(paths["ecg"], ecg_epochs)
        np.save(paths["labels"], labels)
        np.save(paths["fs"], fs)

    def _load_processed(self, sid):
        paths = self._processed_paths(sid)
        ecg_epochs = np.load(paths["ecg"], mmap_mode="r")
        labels = np.load(paths["labels"])
        fs = int(np.load(paths["fs"]))
        return ecg_epochs, labels, fs

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def load_subject(self, idx):
        subject_dir = self.subjects[idx]
        sid = subject_dir.name

        if self._is_processed(sid):
            # 快速路径
            return self._load_processed(sid)

        # 慢路径（只发生一次）
        ecg_epochs, labels, fs = self._load_from_edf(subject_dir)
        self._save_processed(sid, ecg_epochs, labels, fs)

        return ecg_epochs, labels, fs

    def load_all(self):
        data = {}
        for i in range(len(self)):
            ecg_epochs, labels, fs = self.load_subject(i)
            sid = self.subjects[i].name
            data[sid] = {
                "ecg_epochs": ecg_epochs,
                "labels": labels,
                "fs": fs,
            }
        return data
def main():
    data_dir = Path("big/ISRUC/subjects")
    reader = ISRUC_ECG_Reader(data_dir)

    print(f"Total subjects: {len(reader)}")
    print("start time:", time.asctime())

    data = reader.load_all()
    print("end time:", time.asctime())
    print(f"Loaded data for {len(data)} subjects.")

# if __name__ == "__main__":
#     main()
