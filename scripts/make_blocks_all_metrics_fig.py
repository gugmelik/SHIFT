# Figure for the block-group analysis with all metrics (paper_v2, Fig. 4).
# Usage: python scripts/make_blocks_all_metrics_fig.py paper_results/summary.json article_tex/figs/blocks_all_metrics.pdf
import json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
d = json.load(open(sys.argv[1])); R = d['references']; out = sys.argv[2]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42})
cfg = [('blocks_0-3', '1–4'), ('blocks_4-7', '5–8'), ('full', '1–8')]
def val(r, c, m):
    e = R[r]['val_sweep']['6'] if c == 'full' else R[r]['val_ablation'][c]
    return e[m]['mean']
panels = [('dinov2_to_reference', '(а) DINOv2 к референсу ↑', 1),
          ('dinov3_to_reference', '(б) DINOv3 к референсу ↑', 1),
          ('gram_to_reference', '(в) Грам к реф., ×10⁻⁵ ↓', 1e5),
          ('clip_text', '(г) CLIP к запросу ↑', 1),
          ('dinov2_to_origin', '(д) DINOv2 к исх. T2I ↑', 1),
          ('lpips_to_origin', '(е) LPIPS к исх. T2I ↓', 1),
          ('dists_to_origin', '(ж) DISTS к исх. T2I ↓', 1)]
comma = FuncFormatter(lambda v, p: ('%g' % round(v, 4)).replace('.', ','))
fig, axs = plt.subplots(4, 2, figsize=(4.6, 6.6))
x = [0, 1, 2]
for ax, (m, title, k) in zip(axs.flat, panels):
    allv = []
    for r in R:
        y = [val(r, c, m) * k for c, _ in cfg]; allv.append(y)
        ax.plot(x, y, color='#9a9a9a', lw=0.8, marker='o', ms=2.2)
    mean = [sum(v[i] for v in allv) / len(allv) for i in x]
    ax.plot(x, mean, color='black', lw=1.6, marker='o', ms=3.5)
    ax.set_title(title, fontsize=8, loc='left')
    ax.set_xticks(x, [l for _, l in cfg]); ax.set_xlim(-0.25, 2.25)
    ax.yaxis.set_major_formatter(comma)
    ax.grid(axis='y', color='#e3e3e3', lw=0.5); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.set_xlabel('блоки', fontsize=7)
ax = axs.flat[7]
blk = ['blocks_1-2', 'blocks_3-4', 'blocks_5-6', 'blocks_7-8']; xl = ['1–2', '3–4', '5–6', '7–8']
for s, lab, mk in (('img', 'визуальный', 's'), ('txt', 'текстовый', '^')):
    y = [sum(R[r]['token_norms'][b][s] for r in R) / len(R) for b in blk]
    ax.plot(range(4), y, color='black' if s == 'img' else '#7a7a7a', lw=1.4, marker=mk, ms=4)
    ax.annotate(lab, (0, y[0]), xytext=(2, 6 if s=='txt' else 6), textcoords='offset points', fontsize=7)
ax.set_yscale('log'); ax.set_xticks(range(4), xl); ax.set_xlabel('блоки', fontsize=7)
ax.set_title('(з) норма токенов ‖h‖₂', fontsize=8, loc='left')
ax.grid(axis='y', color='#e3e3e3', lw=0.5, which='major'); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(w_pad=1.2, h_pad=0.9)
fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
