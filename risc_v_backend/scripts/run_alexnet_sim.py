import sys
import os

PROJECT_ROOT = 'c:/Mubashir-BTU/Thesis/Codes/Danial/Prune_2'
SIM_DIR = os.path.join(PROJECT_ROOT, 'systolic_sim')
sys.path.insert(0, SIM_DIR)

from simulator import run_both_modes
from metrics import compute_comparison

def simulate_and_print(model_name):
    print(f'==================================================')
    print(f'Simulating model: {model_name}')
    print(f'==================================================')
    results = run_both_modes(model_name=model_name, verbose=False)
    cmp = compute_comparison(results['baseline'], results['packed'])
    
    tb = cmp['totals']['baseline']
    tp = cmp['totals']['packed']
    
    b_cyc = tb['total_cycles']
    p_cyc = tp['total_cycles']
    pct_cyc = (p_cyc - b_cyc) / b_cyc * 100
    
    bw = tb['weight_words_read']
    pw = tp['weight_words_read']
    pct_w = (pw - bw) / bw * 100
    
    ba = tb['act_words_read']
    pa = tp['act_words_read']
    pct_a = (pa - ba) / ba * 100
    
    print(f'Total MACs: {tb["total_macs"]:,}')
    print(f'Baseline Cycles: {b_cyc:,}')
    print(f'Packed Cycles:   {p_cyc:,} ({pct_cyc:+.1f}%)')
    print(f'Weight Words:    Baseline: {bw:,} | Packed: {pw:,} ({pct_w:+.1f}%)')
    print(f'Act Words:       Baseline: {ba:,} | Packed: {pa:,} ({pct_a:+.1f}%)')
    print()

simulate_and_print('AlexNet')
