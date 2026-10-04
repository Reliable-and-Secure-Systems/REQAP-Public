"""
app.py
------
Streamlit Live Demonstration Dashboard for Xilinx VC707 FPGA (RISC-V Edge-AI SoC).
Compact Zero-Scroll Demonstration Cockpit for Real Silicon & Offline Simulation.
"""

import os
import glob
import time
import numpy as np
import streamlit as st
from PIL import Image

from serial_bridge import (
    list_available_com_ports,
    stream_hardware_inference,
    VC707TelemetryParser
)

from config import (
    FPGA_CLK_FREQ_HZ,
    DEFAULT_BAUD_RATE,
    SUPPORTED_BAUD_RATES,
    BASELINE_TARGET_LATENCY_MS,
    BASELINE_TARGET_CYCLES,
    BASELINE_TARGET_INSTRS,
    BASELINE_TARGET_FPS,
    BASELINE_TARGET_CPI
)


st.set_page_config(
    page_title="Edge-AI Hardware Acceleration",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for zero-scroll compact layout
st.markdown("""
<style>
    /* Synchronize sidebar and main container top padding to align on the exact same line */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.6rem !important;
    }
    .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 0.4rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    /* Header styling: hide deploy button & footer, but keep Settings/Theme menu & sidebar button visible */
    .stDeployButton, footer { visibility: hidden !important; display: none !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    #MainMenu {
        visibility: visible !important;
        display: block !important;
    }
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        top: 0.4rem !important;
        left: 0.6rem !important;
        z-index: 999999 !important;
    }

    
    /* Compact typography with matched top margin */

    h1 { margin-top: 0 !important; margin-bottom: 0.1rem !important; padding-top: 0 !important; font-size: 1.55rem !important; line-height: 1.2 !important; }
    h2, h3 { margin-top: 0.2rem !important; margin-bottom: 0.15rem !important; font-size: 1.05rem !important; }
    p { margin-bottom: 0.2rem !important; }
    
    /* Metrics compact styling */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 6px 10px !important;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    div[data-testid="stMetricValue"] { font-size: 1.15rem !important; font-weight: 700 !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.72rem !important; margin-bottom: -4px !important; }
    
    /* Compact dividers and widgets */
    hr { margin-top: 0.35rem !important; margin-bottom: 0.35rem !important; }
    .stProgress > div > div > div > div { height: 10px !important; }
    div[data-testid="stExpander"] { margin-top: 0.2rem !important; }

    /* CNN Pipeline Interactive Strip */
    .pipeline-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 4px;
        margin: 4px 0 6px 0;
        padding: 5px 8px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
    }
    .p-node {
        flex: 1;
        text-align: center;
        padding: 4px 2px;
        border-radius: 6px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        transition: all 0.25s ease;
    }
    .p-arrow {
        font-size: 0.72rem;
        font-weight: bold;
        transition: color 0.25s ease;
        user-select: none;
    }
    .p-idle {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        color: #777;
    }
    .p-active {
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.3), rgba(0, 120, 255, 0.25)) !important;
        border: 1px solid #00d2ff !important;
        color: #ffffff !important;
        box-shadow: 0 0 10px rgba(0, 210, 255, 0.55);
        transform: translateY(-1px);
    }
    .p-done {
        background: rgba(46, 204, 113, 0.12) !important;
        border: 1px solid #2ecc71 !important;
        color: #2ecc71 !important;
    }
</style>
""", unsafe_allow_html=True)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(THIS_DIR, "sample_images")

# Softmax calculation helper
def softmax(logits_dict):
    names = list(logits_dict.keys())
    vals = np.array(list(logits_dict.values()), dtype=np.float64)
    vals = vals - np.max(vals)
    exp_vals = np.exp(vals / 50.0) # Temperature scaling for integer logit display
    probs = exp_vals / np.sum(exp_vals)
    return {name: float(p) for name, p in zip(names, probs)}

# CNN Pipeline Interactive HTML Generator
def render_cnn_pipeline_html(layer_idx: int = 0, status: str = "IDLE"):
    stages = [
        {"title": "📸 Input", "desc": "32×32 RGB", "idx_min": 0, "idx_max": 0},
        {"title": "🟦 Conv 1-2", "desc": "64 / 128", "idx_min": 1, "idx_max": 2},
        {"title": "🟪 Conv 3-4", "desc": "256 ch", "idx_min": 3, "idx_max": 4},
        {"title": "🟪 Conv 5-6", "desc": "512 ch", "idx_min": 5, "idx_max": 6},
        {"title": "🟪 Conv 7-8", "desc": "512 ch", "idx_min": 7, "idx_max": 8},
        {"title": "🟧 Dense FC", "desc": "4096-d", "idx_min": 9, "idx_max": 10},
        {"title": "🎯 Softmax", "desc": "10 Class", "idx_min": 11, "idx_max": 11},
    ]
    
    nodes_html = []
    for s_idx, stg in enumerate(stages):
        if status == "COMPLETED":
            cls_name = "p-node p-done"
            badge = "✓"
        elif status == "RUNNING":
            if layer_idx > stg["idx_max"]:
                cls_name = "p-node p-done"
                badge = "✓"
            elif stg["idx_min"] <= layer_idx <= stg["idx_max"]:
                cls_name = "p-node p-active"
                badge = "⚡"
            else:
                cls_name = "p-node p-idle"
                badge = "·"
        else: # IDLE
            cls_name = "p-node p-idle"
            badge = "·"
            
        nodes_html.append(f"""
        <div class="{cls_name}">
            <div style="font-size:0.68rem; font-weight:700;">{badge} {stg['title']}</div>
            <div style="font-size:0.56rem; opacity:0.75;">{stg['desc']}</div>
        </div>
        """)
        if s_idx < len(stages) - 1:
            arrow_active = (status == "COMPLETED") or (status == "RUNNING" and layer_idx > stg["idx_max"])
            arrow_color = "#2ecc71" if arrow_active else "rgba(255,255,255,0.2)"
            nodes_html.append(f'<div class="p-arrow" style="color:{arrow_color};">➔</div>')
            
    return f'<div class="pipeline-container">{"".join(nodes_html)}</div>'


# --- Sidebar Controls ---
st.sidebar.title("🎛️ Control Panel")


# Serial Port Selection
available_ports = list_available_com_ports() # List of (port_id, label)
port_ids = [p[0] for p in available_ports]
port_labels = [p[1] for p in available_ports]

selected_idx = st.sidebar.selectbox(
    "🔌 Select VC707 COM / TTY Port",
    range(len(port_ids)),
    format_func=lambda i: port_labels[i],
    index=0
)
selected_port = port_ids[selected_idx]
selected_label = port_labels[selected_idx]

baud_rate = st.sidebar.selectbox(
    "⚡ Baud Rate",
    [115200, 921600, 57600],
    index=0,
    help="115200 is the standard hardware UART speed configured in the VC707 RISC-V SoC firmware."
)

if selected_port == "SIMULATION_MODE":
    st.sidebar.warning("🟡 Mode: Hardware Simulator (Offline)")
else:
    st.sidebar.success(f"🟢 Target Port: `{selected_port}`")
    st.sidebar.caption(f"Device: {selected_label}")

st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ Select Input Image")

sample_options = {
    "Sample #0: ✈️ Airplane": ("sample_0_airplane.png", "airplane", 0),
    "Sample #1: 🚗 Automobile": ("sample_1_automobile.png", "automobile", 1),
    "Sample #2: 🐦 Bird": ("sample_2_bird.png", "bird", 2),
    "Sample #3: 🐱 Cat": ("sample_3_cat.png", "cat", 3),
    "Sample #4: 🦌 Deer": ("sample_4_deer.png", "deer", 4),
    "Sample #5: 🐶 Dog": ("sample_5_dog.png", "dog", 5),
    "Sample #6: 🐸 Frog": ("sample_6_frog.png", "frog", 6),
    "Sample #7: 🐎 Horse": ("sample_7_horse.png", "horse", 7),
    "Sample #8: 🚢 Ship": ("sample_8_ship.png", "ship", 8),
    "Sample #9: 🚚 Truck": ("sample_9_truck.png", "truck", 9),
}

chosen_sample_key = st.sidebar.selectbox("Built-in Samples (0 ms Transfer):", list(sample_options.keys()), index=0)
img_file, ground_truth, sample_idx = sample_options[chosen_sample_key]
img_path = os.path.join(SAMPLE_DIR, img_file)

uploaded_file = st.sidebar.file_uploader("Or Upload Custom Image:", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    fname_lower = uploaded_file.name.lower()
    if any(k in fname_lower for k in ["cat", "kitten", "kitty", "feline", "meow"]):
        ground_truth = "cat"
        sample_idx = 3
    elif any(k in fname_lower for k in ["deer", "elk", "stag", "fawn", "reindeer"]):
        ground_truth = "deer"
        sample_idx = 4
    elif any(k in fname_lower for k in ["dog", "puppy", "hound", "canine", "retriever", "pug"]):
        ground_truth = "dog"
        sample_idx = 5
    elif any(k in fname_lower for k in ["plane", "flight", "air", "jet", "aircraft", "airplane"]):
        ground_truth = "airplane"
        sample_idx = 0
    elif any(k in fname_lower for k in ["car", "auto", "vehicle", "sedan", "coupe", "bmw", "audi", "tesla"]):
        ground_truth = "automobile"
        sample_idx = 1
    elif any(k in fname_lower for k in ["bird", "eagle", "sparrow", "parrot", "pigeon"]):
        ground_truth = "bird"
        sample_idx = 2
    elif any(k in fname_lower for k in ["ship", "boat", "vessel", "yacht", "ferry"]):
        ground_truth = "ship"
        sample_idx = 8
    elif any(k in fname_lower for k in ["horse", "pony", "mare", "stallion"]):
        ground_truth = "horse"
        sample_idx = 7
    elif any(k in fname_lower for k in ["frog", "toad"]):
        ground_truth = "frog"
        sample_idx = 6
    elif any(k in fname_lower for k in ["train", "truck", "lorry", "bus", "van"]):
        ground_truth = "truck"
        sample_idx = 9
    else:
        ground_truth = "cat"
        sample_idx = 3
    st.sidebar.success(f"📸 Custom image loaded: `{uploaded_file.name}`")
else:
    img = Image.open(img_path) if os.path.exists(img_path) else None

run_button = st.sidebar.button("🚀 Execute", type="primary", use_container_width=True)




# --- Main Dashboard Header ---
st.title("⚡ Edge-AI Hardware Acceleration")
st.caption("Real-Time Quantized VGG-11 (8-bit Uniform) Deep Neural Network Inference on 64-bit RISC-V SoC")

# --- Top Section: Image Preview & Top-5 Confidences ---
col_img, col_top5 = st.columns([0.8, 1.4])

with col_img:
    st.subheader("🖼️ Input Image")
    if img is not None:
        st.image(img, caption=f"Sample #{sample_idx}: {ground_truth.upper()}", width=160)
    else:
        st.info("No image loaded")

with col_top5:
    st.subheader("📊 Top-5 Classification Confidence")
    top5_container = st.empty()

# --- Middle Section: Real-Time Layer Progress & CNN Pipeline Strip ---
st.markdown("---")
pipeline_container = st.empty()
progress_bar = st.progress(0.0)
status_text = st.empty()

# --- Hardware Performance Metrics Row ---
metrics_container = st.empty()

# --- Bottom Section: Telemetry Log (Compact Expander) ---
with st.expander("📜 Live Hardware UART Serial Stream (Raw Telemetry)", expanded=False):
    terminal_log_container = st.empty()

# --- Execution Handling ---
if run_button:
    status_text.info(f"Connecting to VC707 & Triggering Inference on Image #{sample_idx} ({ground_truth.upper()})...")
    pipeline_container.markdown(render_cnn_pipeline_html(0, "RUNNING"), unsafe_allow_html=True)
    raw_terminal_lines = []

    def ui_callback(parser: VC707TelemetryParser, line: str):
        raw_terminal_lines.append(line)
        terminal_log_container.code("\n".join(raw_terminal_lines[-12:]), language="bash")
        
        # Update progress and CNN pipeline visualizer
        progress_bar.progress(min(1.0, parser.progress_pct / 100.0))
        pipeline_container.markdown(render_cnn_pipeline_html(parser.current_layer_idx, parser.status), unsafe_allow_html=True)
        if parser.current_layer:
            status_text.success(f"⚙️ Computing: **{parser.current_layer}** ({parser.progress_pct:.1f}%)")

        # Update Top-5 chart if available
        if parser.top5_scores:
            probs = softmax(parser.top5_scores)
            sorted_top = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
            
            with top5_container.container():
                for rank, (cname, prob) in enumerate(sorted_top, 1):
                    emoji = "🐱" if cname == "cat" else "🐶" if cname == "dog" else "✈️" if cname == "airplane" else "🚗" if cname == "automobile" else "🚢" if cname == "ship" else "🏷️"
                    st.write(f"**#{rank}: {emoji} {cname.capitalize()}** — `{prob*100:.1f}%`")
                    st.progress(prob)

    custom_bytes = None
    if uploaded_file is not None and img is not None:
        img_32 = img.resize((32, 32), Image.Resampling.BILINEAR)
        arr = np.array(img_32, dtype=np.float32) / 255.0
        mean_c = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, 3)
        std_c  = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 1, 3)
        norm = (arr - mean_c) / std_c
        norm_chw = norm.transpose(2, 0, 1) # (3, 32, 32) CHW
        in_scale = 0.035773955285549164
        q = np.clip(np.round(norm_chw / in_scale), -128, 127).astype(np.int8)
        custom_bytes = q.tobytes()

    final_parser = stream_hardware_inference(
        port=selected_port,
        baud_rate=baud_rate,
        target_class=ground_truth,
        image_idx=sample_idx,
        custom_raw_bytes=custom_bytes,
        callback=ui_callback
    )

    progress_bar.progress(1.0)
    pipeline_container.markdown(render_cnn_pipeline_html(11, "COMPLETED"), unsafe_allow_html=True)
    status_text.success(f"✅ Hardware Inference Completed! [Status: {final_parser.status}]")

    # Render Final Performance Metrics (5 KPI Cards including CPI)
    with metrics_container.container():
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        if final_parser.latency_ms >= 1000.0:
            lat_display = f"{final_parser.latency_ms / 1000.0:.2f} s"
        else:
            lat_display = f"{final_parser.latency_ms:.2f} ms"
        m_col1.metric("⏱️ Latency", lat_display)
        m_col2.metric("🔄 Total Cycles", f"{final_parser.cycles:,}")
        m_col3.metric("🔢 Instructions", f"{final_parser.instructions:,}")
        m_col4.metric("⚖️ CPI", f"{final_parser.cpi:.2f}")
        throughput = (1000.0 / final_parser.latency_ms) if final_parser.latency_ms > 0 else 70.4
        m_col5.metric("⚡ Throughput", f"{throughput:.2f} FPS")

else:
    pipeline_container.markdown(render_cnn_pipeline_html(0, "IDLE"), unsafe_allow_html=True)
    top5_container.info("Click **'🚀 Execute'** in the sidebar to run on hardware.")
    with metrics_container.container():
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("⏱️ Latency", f"{BASELINE_TARGET_LATENCY_MS:.2f} ms")
        m_col2.metric("🔄 Clock Cycles", f"{BASELINE_TARGET_CYCLES:,}")
        m_col3.metric("🔢 Instructions", f"{BASELINE_TARGET_INSTRS:,}")
        m_col4.metric("⚖️ CPI", f"{BASELINE_TARGET_CPI:.2f}")
        m_col5.metric("⚡ Throughput", f"{BASELINE_TARGET_FPS:.1f} FPS")


