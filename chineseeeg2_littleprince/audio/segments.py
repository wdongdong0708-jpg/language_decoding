from __future__ import annotations

import bisect
import re
import wave
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import json

# XLSX 文件的 XML 命名空间，用于底层手动解析 Excel
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
XLSX_RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


@dataclass(frozen=True)
class AudioSegment:
    """存储最终切分出来的音频片段元数据"""
    speaker_id: str             # 说话人 ID
    text_embedding_idx: int     # 文本嵌入的索引
    audio_event_idx: int        # 音频事件的索引
    event_time_scale: float     # events_data 时间到 WAV 实际时间的缩放因子
    text: str                   # 对应的文本内容
    audio_file_path: Path       # 音频文件路径
    audio_start_time: float     # 在该音频文件中的起始时间（秒）
    audio_stop_time: float      # 在该音频文件中的结束时间（秒）
    audio_sample_rate: int      # 音频采样率
    audio_start_sample: int     # 起始采样点位置
    audio_stop_sample: int      # 结束采样点位置
    n_audio_samples: int        # 总采样点数


@dataclass(frozen=True)
class WavInfo:
    """存储 WAV 音频文件的基本信息"""
    path: Path                  # 文件路径
    sample_rate: int            # 采样率
    n_samples: int              # 总帧数/采样点数


def _audio_index(path: Path) -> int:
    """
    辅助函数：从音频文件名（如 'audio_12.wav'）中提取出数字序号
    用于对音频文件进行正确排序
    """
    match = re.fullmatch(r"audio_(\d+)", path.stem)
    if not match:
        raise ValueError(f"无法从路径中解析音频序号: {path}")
    return int(match.group(1))


def _wav_info(path: Path) -> WavInfo:
    """
    辅助函数：读取指定路径的 WAV 文件，获取其采样率和总样本数
    """
    with wave.open(str(path), "rb") as wav:
        return WavInfo(path=path, sample_rate=wav.getframerate(), n_samples=wav.getnframes())


def read_xlsx_column(path: str | Path, column: str = "A") -> list[str]:
    """
    为了避免依赖 pandas/openpyxl 等大库，通过解压 zip 的方式手动读取 Excel (.xlsx) 指定列的所有文本
    """
    path = Path(path)
    column = column.upper()
    pattern = re.compile(rf"{re.escape(column)}\d+")  # 匹配形如 A1, A2, A3 的单元格坐标
    
    with zipfile.ZipFile(path) as zf:
        shared_strings = []
        # 1. 解析 Excel 的共享字符串表（Excel 为了节省空间，把重复文本放在这里）
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall(XLSX_NS + "si"):
                shared_strings.append("".join((text.text or "") for text in item.iter(XLSX_NS + "t")))

        # 2. 找到第一个工作表（Sheet）的文件路径
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        sheet = workbook.find(XLSX_NS + "sheets")[0]
        target = rel_map[sheet.attrib[XLSX_RNS + "id"]]
        if not target.startswith("xl/"):
            target = "xl/" + target

        # 3. 遍历该工作表的所有行，提取指定列的数据
        sheet_xml = ET.fromstring(zf.read(target))
        values = []
        for row in sheet_xml.findall(".//" + XLSX_NS + "sheetData/" + XLSX_NS + "row"):
            value = ""
            for cell in row.findall(XLSX_NS + "c"):
                # 如果当前单元格坐标匹配我们想要的列（例如 A 列）
                if pattern.fullmatch(cell.attrib.get("r", "")):
                    cell_value = cell.find(XLSX_NS + "v")
                    if cell_value is not None:
                        # 如果类型是 "s"，说明是共享字符串，去 shared_strings 查表
                        value = (
                            shared_strings[int(cell_value.text)]
                            if cell.attrib.get("t") == "s"
                            else cell_value.text
                        )
                    break
            values.append(str(value).strip())
        return values


class AudioTimeline:
    """将数据集中句子的‘全局时间戳’映射到具体的 WAV 文件窗口中"""

    def __init__(
        self,
        speaker_id: str,
        audio_dir: str | Path,
        chapter_start_ms: list[int],
        row_start_ms: list[int],
        row_stop_ms: list[int],
        wavs: list[WavInfo],
    ):
        # 验证输入数据的长度是否对齐
        if len(row_start_ms) != len(row_stop_ms):
            raise ValueError(
                f"{speaker_id} 的 ROWS/ROWE 长度不匹配: "
                f"{len(row_start_ms)} != {len(row_stop_ms)}"
            )
        if len(chapter_start_ms) != len(wavs):
            raise ValueError(
                f"{speaker_id} 的章节数与音频文件数不匹配: "
                f"{len(chapter_start_ms)} != {len(wavs)}"
            )
        self.speaker_id = speaker_id
        self.audio_dir = Path(audio_dir)
        self.chapter_start_ms = chapter_start_ms  # 每个音频文件在全局时间线上的起始时间(毫秒)
        self.row_start_ms = row_start_ms          # 每个音频事件的起始时间(毫秒)
        self.row_stop_ms = row_stop_ms            # 每个音频事件的结束时间(毫秒)
        self.wavs = wavs                          # 包含的所有音频文件信息列表

    @classmethod
    def from_directory(cls, speaker_id: str, audio_dir: str | Path) -> "AudioTimeline":
        """
        工厂方法：从指定目录加载 events_data.json 并扫描音频文件，构建 AudioTimeline 实例
        """
        audio_dir = Path(audio_dir)
        # 读取包含时间戳的 json 文件
        with (audio_dir / "events_data.json").open("r", encoding="utf-8") as f:
            events = json.load(f)
        
        # 扫描目录下所有的 audio_*.wav 文件，并按数字序号排序，获取它们的信息
        wavs = [_wav_info(path) for path in sorted(audio_dir.glob("audio_*.wav"), key=_audio_index)]
        
        return cls(
            speaker_id=speaker_id,
            audio_dir=audio_dir,
            chapter_start_ms=[int(value) for value in events["begn_nonzero_indices"]],
            row_start_ms=[int(value) for value in events["ROWS_times"]],
            row_stop_ms=[int(value) for value in events["ROWE_times"]],
            wavs=wavs,
        )

    def segment_for_text_embedding(
        self,
        text_embedding_idx: int,
        text: str = "",
        event_offset: int = 1,
        event_time_scale: float = 1.0,
        end_tolerance_seconds: float = 0.0,
    ) -> AudioSegment:
        """
        核心方法：根据文本嵌入的索引，计算并返回其在具体音频文件中对应的切片区域（AudioSegment）
        """
        if event_time_scale <= 0:
            raise ValueError(f"event_time_scale must be positive, got {event_time_scale}")
        if end_tolerance_seconds < 0:
            raise ValueError(
                f"end_tolerance_seconds must be non-negative, got {end_tolerance_seconds}"
            )

        # 1. 计算实际对应的音频事件索引
        audio_event_idx = int(text_embedding_idx) + event_offset
        if audio_event_idx < 0 or audio_event_idx >= len(self.row_start_ms):
            raise IndexError(
                f"audio_event_idx={audio_event_idx} 超出事件范围 0..{len(self.row_start_ms) - 1}"
            )

        # 2. 获取该事件在全局时间线上的绝对起始和结束时间
        start_ms = self.row_start_ms[audio_event_idx]
        stop_ms = self.row_stop_ms[audio_event_idx]
        if stop_ms <= start_ms:
            raise ValueError(f"无效的音频时间窗口 {audio_event_idx}: {start_ms}..{stop_ms}")

        # 3. 使用二分查找，确定这段全局时间落在哪一个音频文件（章节）内
        chapter_idx = bisect.bisect_right(self.chapter_start_ms, start_ms) - 1
        if chapter_idx < 0:
            raise ValueError(f"事件 {audio_event_idx} 在第一章开始之前: {start_ms}")

        # 4. 将全局毫秒时间转换为该音频文件内部的相对时间
        chapter_start_ms = self.chapter_start_ms[chapter_idx]
        wav = self.wavs[chapter_idx]
        # Little Prince 的 events_data 时间来自原始 1000 Hz EEG，而当前 PL
        # manifest 使用重采样后的 250 Hz EEG。调用方通过 event_time_scale=4
        # 将事件时间恢复到 WAV 的实际时间轴。
        local_start_ms = (start_ms - chapter_start_ms) * event_time_scale
        local_stop_ms = (stop_ms - chapter_start_ms) * event_time_scale
        
        # 5. 将相对时间（毫秒）转换为音频采样的具体具体位置（Sample Index）
        start_sample = int(round(local_start_ms * wav.sample_rate / 1000.0))
        stop_sample = int(round(local_stop_ms * wav.sample_rate / 1000.0))

        # 每章最后一个 ROWE 可能因事件取整比 WAV 末尾多出少量采样点。
        # 仅在显式容差范围内裁到 WAV 末尾，避免掩盖真正的时间轴错误。
        max_end_overrun = int(round(end_tolerance_seconds * wav.sample_rate))
        if stop_sample > wav.n_samples and stop_sample - wav.n_samples <= max_end_overrun:
            stop_sample = wav.n_samples
            local_stop_ms = stop_sample * 1000.0 / wav.sample_rate
        
        # 边界安全性检查
        if start_sample < 0 or stop_sample <= start_sample or stop_sample > wav.n_samples:
            raise ValueError(
                f"音频事件 {audio_event_idx} 映射到了文件 {wav.path} 外部: "
                f"{start_sample}:{stop_sample}，文件总长 {wav.n_samples}"
            )

        # 6. 返回组装好的音频切片元数据对象
        return AudioSegment(
            speaker_id=self.speaker_id,
            text_embedding_idx=int(text_embedding_idx),
            audio_event_idx=audio_event_idx,
            event_time_scale=float(event_time_scale),
            text=text,
            audio_file_path=wav.path,
            audio_start_time=local_start_ms / 1000.0,
            audio_stop_time=local_stop_ms / 1000.0,
            audio_sample_rate=wav.sample_rate,
            audio_start_sample=start_sample,
            audio_stop_sample=stop_sample,
            n_audio_samples=stop_sample - start_sample,
        )


def littleprince_speaker_for_subject(subject: str) -> str:
    """
    根据小王子数据集的被试标号（如 'sub-01'），映射出对应的说话人 ID（发音人是男是女）
    - 被试 1~4 对应小王子女性发音人 1
    - 被试 5~8 对应小王子男性发音人 1
    """
    match = re.fullmatch(r"sub-(\d+)", subject)
    if not match:
        raise ValueError(f"无法从小王子被试标号中解析 ID: {subject!r}")
    subject_id = int(match.group(1))
    if 1 <= subject_id <= 4:
        return "littleprince_f1"
    if 5 <= subject_id <= 8:
        return "littleprince_m1"
    raise ValueError(f"未找到该被试 {subject} 对应的听觉任务发音人规则")
