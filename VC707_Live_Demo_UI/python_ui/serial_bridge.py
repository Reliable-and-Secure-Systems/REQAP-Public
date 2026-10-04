"""
serial_bridge.py
----------------
Hardware UART Bridge for Xilinx VC707 FPGA (RISC-V SoC).
Handles COM port auto-discovery, live stream parsing, and offline hardware simulation.
"""

import time
import re
import numpy as np
from typing import Dict, List, Optional, Callable

from config import FPGA_CLK_FREQ_HZ, DEFAULT_BAUD_RATE, CIFAR10_CLASSES



try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


def list_available_com_ports() -> List[tuple]:
    """Scans and returns all connected serial / COM ports with descriptions."""
    results = [("SIMULATION_MODE", "🟡 SIMULATION_MODE (Offline Presentation)")]
    if HAS_SERIAL:
        for p in serial.tools.list_ports.comports():
            # e.g., ("COM5", "COM5 - Standard Serial over Bluetooth link")
            desc = f"{p.device} - {p.description}"
            results.append((p.device, desc))
    return results



class VC707TelemetryParser:
    """Parses structured ASCII tokens from the VC707 UART stream."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.model = "VGG-11 (8-bit)"
        self.mode = "8-bit Uniform Baseline"
        self.current_layer = ""
        self.current_layer_idx = 0
        self.progress_pct = 0.0
        self.cycles = 0
        self.instructions = 0
        self.cpi = 0.0
        self.latency_ms = 0.0
        self.top5_scores: Dict[str, float] = {}
        self.pred_class_id = -1
        self.pred_class_name = ""
        self.status = "IDLE"
        self.raw_logs: List[str] = []

    def parse_line(self, line: str) -> Optional[str]:
        """Parses a single line of UART telemetry."""
        line = line.strip()
        if not line:
            return None
        self.raw_logs.append(line)

        if line.startswith("[START]"):
            self.status = "RUNNING"
            self.progress_pct = 5.0
            self.current_layer_idx = 0
            return "START"

        elif line.startswith("[LAYER]"):
            # e.g., [LAYER] idx=1 name=Conv2d_1 progress=12.5%
            m_idx = re.search(r"idx=([0-9]+)", line)
            m_name = re.search(r"name=([A-Za-z0-9_]+)", line)
            m_prog = re.search(r"progress=([0-9.]+)%", line)
            if m_idx:
                self.current_layer_idx = int(m_idx.group(1))
            if m_name:
                self.current_layer = m_name.group(1)
            if m_prog:
                self.progress_pct = float(m_prog.group(1))
            return "LAYER"


        elif line.startswith("[METRIC]"):
            # e.g., [METRIC] cycles=1420500 instrs=1120400 time_ms=14.20
            m_cyc = re.search(r"cycles=([0-9]+)", line)
            m_ins = re.search(r"instrs=([0-9]+)", line)
            m_tim = re.search(r"time_ms=([0-9.]+)", line)
            if m_cyc: self.cycles = int(m_cyc.group(1))
            if m_ins: self.instructions = int(m_ins.group(1))
            if m_tim: 
                self.latency_ms = float(m_tim.group(1))
            elif self.cycles > 0:
                self.latency_ms = (self.cycles / FPGA_CLK_FREQ_HZ) * 1000.0

            if self.instructions > 0:
                self.cpi = self.cycles / self.instructions
            else:
                self.cpi = 1.27
            return "METRIC"



        elif line.startswith("[TOP5]"):
            # e.g., [TOP5] 0:airplane=-28 1:automobile=-270 3:cat=416 ...
            tokens = line.replace("[TOP5]", "").strip().split()
            scores = {}
            for tok in tokens:
                if "=" in tok:
                    cls_part, score_part = tok.split("=")
                    if ":" in cls_part:
                        _, cname = cls_part.split(":")
                    else:
                        cname = cls_part
                    scores[cname] = float(score_part)
            self.top5_scores = scores
            return "TOP5"

        elif line.startswith("[PRED]"):
            # e.g., [PRED] class_id=3 class_name=cat status=PASS
            m_id = re.search(r"class_id=([0-9]+)", line)
            m_name = re.search(r"class_name=([A-Za-z0-9_]+)", line)
            if m_id: self.pred_class_id = int(m_id.group(1))
            if m_name: self.pred_class_name = m_name.group(1)
            return "PRED"

        elif line.startswith("[DONE]"):
            self.status = "COMPLETED"
            self.progress_pct = 100.0
            return "DONE"

        return "LOG"


def stream_hardware_inference(
    port: str,
    baud_rate: int = 115200,
    target_class: str = "cat",
    image_idx: int = 0,
    custom_raw_bytes: Optional[bytes] = None,
    callback: Optional[Callable[[VC707TelemetryParser, str], None]] = None
) -> VC707TelemetryParser:
    """
    Connects to the VC707 COM port or runs high-fidelity hardware simulation if in SIMULATION_MODE.
    Calls `callback(parser, line)` as each telemetry line arrives.
    """
    parser = VC707TelemetryParser()

    if port == "SIMULATION_MODE" or not HAS_SERIAL:
        # Realistic class logits generation based on selected sample
        cifar_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        t_cls = target_class.lower() if target_class.lower() in cifar_classes else "cat"
        t_idx = cifar_classes.index(t_cls)

        # Generate realistic high-confidence logits for the recognized target class
        logits_str_list = []
        for i, c in enumerate(cifar_classes):
            if i == t_idx:
                score = 420 + (i * 7) % 30
            else:
                score = -50 + (i * 23) % 80
            logits_str_list.append(f"{i}:{c}={score}")
        top5_str = " ".join(logits_str_list)


        # Realistic hardware simulation stream with slight realistic jitter per sample
        base_cyc = 1420500 + (t_idx * 1420)
        base_ins = 1120400 + (t_idx * 980)
        base_tim = (base_cyc / FPGA_CLK_FREQ_HZ) * 1000.0 # Configurable clock time in ms



        sim_stream = [
            "==================================================",
            "  Xilinx VC707 FPGA VGG-11 Real-Time Inference Engine",
            "==================================================",
            f"[START] model=VGG-11 mode=8bit_baseline image_idx={image_idx} image_name={t_cls}",
            "[LAYER] idx=1 name=Conv2d_1 progress=12.5%",
            "[LAYER_DONE] idx=1 name=Conv2d_1",
            "[LAYER] idx=2 name=Conv2d_2 progress=25.0%",
            "[LAYER_DONE] idx=2 name=Conv2d_2",
            "[LAYER] idx=3 name=Conv2d_3 progress=37.5%",
            "[LAYER_DONE] idx=3 name=Conv2d_3",
            "[LAYER] idx=4 name=Conv2d_4 progress=50.0%",
            "[LAYER_DONE] idx=4 name=Conv2d_4",
            "[LAYER] idx=5 name=Conv2d_5 progress=62.5%",
            "[LAYER_DONE] idx=5 name=Conv2d_5",
            "[LAYER] idx=6 name=Conv2d_6 progress=75.0%",
            "[LAYER_DONE] idx=6 name=Conv2d_6",
            "[LAYER] idx=7 name=Conv2d_7 progress=87.5%",
            "[LAYER_DONE] idx=7 name=Conv2d_7",
            "[LAYER] idx=8 name=Conv2d_8 progress=92.0%",
            "[LAYER_DONE] idx=8 name=Conv2d_8",
            "[LAYER] idx=9 name=Classifier_FC1 progress=95.0%",
            "[LAYER_DONE] idx=9 name=Classifier_FC1",
            "[LAYER] idx=10 name=Classifier_FC2 progress=98.0%",
            "[LAYER_DONE] idx=10 name=Classifier_FC2",
            "[LAYER] idx=11 name=Classifier_Head progress=100.0%",
            "[LAYER_DONE] idx=11 name=Classifier_Head",
            f"[METRIC] cycles={base_cyc} instrs={base_ins} time_ms={base_tim:.2f}",
            f"[TOP5] {top5_str}",
            f"[PRED] class_id={t_idx} class_name={t_cls} status=PASS",
            "[DONE]"
        ]
        for line in sim_stream:
            time.sleep(0.06)  # Mimic realistic FPGA layer execution timing
            parser.parse_line(line)
            if callback:
                callback(parser, line)
        return parser

    # Real Hardware Serial Connection
    try:
        with serial.Serial(port, baud_rate, timeout=4.0) as ser:
            ser.reset_input_buffer()
            if custom_raw_bytes is not None and len(custom_raw_bytes) == 3072:
                # Stream custom 3072 image bytes to FPGA
                ser.write(b"U\n")
                time.sleep(0.05)
                ser.write(custom_raw_bytes)
                ser.flush()
            else:
                # Send chosen image index (0-4) to trigger FPGA execution
                cmd_bytes = f"{image_idx}\n".encode("utf-8")
                ser.write(cmd_bytes)
                ser.flush()
                time.sleep(0.05)

            start_wait = time.time()
            while time.time() - start_wait < 12.0:
                raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
                if raw_line:
                    event = parser.parse_line(raw_line)
                    if callback:
                        callback(parser, raw_line)
                    if event == "DONE":
                        break
    except Exception as e:
        parser.raw_logs.append(f"[ERROR] Serial Communication Failed: {str(e)}")
        parser.raw_logs.append("⚠️ TIP: If using Linux or Windows, ensure Minicom, PuTTY, or TeraTerm is completely CLOSED so Python can access the port.")
        parser.status = "PORT_LOCKED_OR_OFFLINE"

    return parser



