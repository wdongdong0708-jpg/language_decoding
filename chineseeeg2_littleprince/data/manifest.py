"""
💡 核心流程关系
外部通常先调用 load_manifest("your_file.csv")。

load_manifest 内部依靠 _record_from_row 将 CSV 的纯文本行包装成一个个有类型的 ManifestRecord 结构体。

加载完成后，将得到的 list[ManifestRecord] 传入 validate_manifest 进行数据逻辑与磁盘文件存在性的双重体检，确保后续的深度学习训练或数据处理不会中途因找不到文件而报错。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestRecord:
    """
    存储单条清单记录的数据类（不可变对象）。
    用于映射 EEG 信号分片、实验设计（被试、任务、Run）以及对应的文本嵌入（Embedding）元数据。
    """
    subject: str               # 被试编号（例如 'sub-01'）
    session: str               # 实验场次/课时（例如 'ses-01'）
    task: str                  # 实验任务名称（例如 'listening'）
    run: int                   # 当前任务的执行轮次（Run 序号）
    local_row_idx: int         # 当前 Run 内的局部行索引
    global_row_idx: int        # 跨越所有数据时的全局行索引
    text_embedding_idx: int    # 对应文本嵌入特征的索引
    label_id: int              # 标签 ID（通常用于分类或对齐，默认同 text_embedding_idx）
    start_time: float          # 当前片段在音频/实验中的起始时间（秒）
    stop_time: float           # 当前片段在音频/实验中的结束时间（秒）
    sfreq: float               # EEG 信号的采样频率（Sampling Frequency，如 500.0 Hz）
    start_sample: int          # EEG 数据的起始采样点索引（Sample Index）
    stop_sample: int           # EEG 数据的结束采样点索引
    n_samples: int             # 当前片段包含的总采样点数（信号长度）
    eeg_vhdr_path: Path        # BrainVision 格式 EEG 头文件 (.vhdr) 的存储路径
    events_tsv_path: Path       # 实验事件文件 (.tsv) 的存储路径
    text_embedding_path: Path  # 对应文本嵌入向量文件（如 .npy 或 .pt）的存储路径


def _record_from_row(row: dict[str, str]) -> ManifestRecord:
    """
    辅助函数：将 CSV 读取出来的单行字典数据（全是字符串）
    转换为强类型的 ManifestRecord 实例，并处理相应的数据类型转换。
    """
    return ManifestRecord(
        subject=row["subject"],
        session=row["session"],
        task=row["task"],
        run=int(row["run"]),
        local_row_idx=int(row["local_row_idx"]),
        global_row_idx=int(row["global_row_idx"]),
        text_embedding_idx=int(row["text_embedding_idx"]),
        # 如果 CSV 中没有配置 label_id 或为空，则默认复用 text_embedding_idx
        label_id=int(row.get("label_id") or row["text_embedding_idx"]),
        start_time=float(row["start_time"]),
        stop_time=float(row["stop_time"]),
        sfreq=float(row["sfreq"]),
        start_sample=int(row["start_sample"]),
        stop_sample=int(row["stop_sample"]),
        n_samples=int(row["n_samples"]),
        eeg_vhdr_path=Path(row["eeg_vhdr_path"]),
        events_tsv_path=Path(row["events_tsv_path"]),
        text_embedding_path=Path(row["text_embedding_path"]),
    )


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    """
    从指定的 CSV 清单文件中加载所有记录。
    
    参数:
        path: CSV 文件的路径（支持字符串或 Path 对象）
    返回:
        包含所有解析后的 ManifestRecord 对象的列表
    异常:
        ValueError: 如果清单文件内容为空则抛出此异常
    """
    manifest_path = Path(path)
    # 以 UTF-8 编码打开 CSV 文件，newline="" 是标准库 csv 模块的标准推荐写法
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        # 使用 csv.DictReader 将每一行读取为字典，并批量转换为 ManifestRecord 对象
        records = [_record_from_row(row) for row in csv.DictReader(f)]
        
    # 如果文件里没有任何记录，抛出异常阻断后续逻辑
    if not records:
        raise ValueError(f"清单文件内容为空: {manifest_path}")
        
    return records


def validate_manifest(records: list[ManifestRecord]) -> None:
    """
    对加载进来的清单记录列表进行鲁棒性校验。
    检查项包括：采样窗口的数学逻辑、总样本数是否匹配、相关的实体文件在磁盘上是否存在。
    
    参数:
        records: 待校验的 ManifestRecord 列表
    异常:
        ValueError: 采样点计算逻辑错误时抛出
        FileNotFoundError: 关键的 EEG 文件或文本嵌入文件在磁盘上不存在时抛出
    """
    for record in records:
        # 1. 检查结束采样点是否大于起始采样点
        if record.stop_sample <= record.start_sample:
            raise ValueError(f"记录 {record.global_row_idx} 中存在无效的采样窗口 (stop <= start)")
            
        # 2. 检查记录的总样本数 (n_samples) 是否真的等于 (stop_sample - start_sample)
        if record.n_samples != record.stop_sample - record.start_sample:
            raise ValueError(f"记录 {record.global_row_idx} 中的 n_samples 与实际采样区间差值不匹配")
            
        # 3. 检查脑电头文件 (.vhdr) 是否切实在磁盘中存在（防止后面训练或读取时崩溃）
        if not record.eeg_vhdr_path.exists():
            raise FileNotFoundError(f"找不到脑电文件: {record.eeg_vhdr_path}")
            
        # 4. 检查对应的文本嵌入向量文件是否存在
        if not record.text_embedding_path.exists():
            raise FileNotFoundError(f"找不到文本嵌入文件: {record.text_embedding_path}")