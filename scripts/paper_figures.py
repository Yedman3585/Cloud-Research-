"""Figures for the paper from the result files (PDF for the paper, PNG for previews).

    python scripts/paper_figures.py

Reads artifacts/smokescreen-ifogsim/runs.jsonl (final comparison in iFogSim) and
artifacts/robustness/runs.jsonl (robustness and ablations); writes docs/figures/.
Error bars are 95 % Student-t half-widths over seeds.
"""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures'
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

# Categorical slots in fixed order (validated: scripts in the data-viz method); the
# method takes slot 1. Text stays in ink colours, never series colours.
COLOR = {'ps-anneal-alerts': '#2a78d6', 'value-greedy-alerts': '#eb6834', 'greedy-v2': '#1baf7a',
         'ps-anneal': '#8a8984'}
MARKER = {'ps-anneal-alerts': 'o', 'value-greedy-alerts': 's', 'greedy-v2': '^', 'ps-anneal': 'D'}
LABEL = {'ps-anneal-alerts': 'ps-anneal + alerts (ours)', 'value-greedy-alerts': 'value-greedy + alerts',
         'greedy-v2': 'greedy-v2 (best rule)', 'ps-anneal': 'ps-anneal, no alerts'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def mean_ci(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return (values[0] if values else float('nan')), 0.0
    return statistics.fmean(values), T95.get(len(values) - 1, 1.96) * statistics.stdev(values) / len(values) ** 0.5


def load(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, width=0.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main_results(plt):
    rows = load(ROOT / 'artifacts/smokescreen-ifogsim/runs.jsonl')
    order = ['greedy-v2', 'value-greedy-alerts', 'ps-anneal', 'ps-anneal-alerts']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={'width_ratios': [1.2, 1, 1]})
    for ax, key, title in zip(axes, ('objective', 'missed_quiet', 'missed_flood'),
                              ('Objective (lower is better)', 'Missed low-rate attack tasks',
                               'Missed flood tasks')):
        for k, p in enumerate(order):
            m, h = mean_ci([r[key] for r in rows if r['policy'] == p])
            ax.bar(k, m, width=0.62, color=COLOR[p], edgecolor='white', linewidth=1.5, zorder=2)
            ax.errorbar(k, m, yerr=h, color=INK, capsize=2.5, linewidth=0.8, zorder=3)
            ax.text(k, m + h + 0.01, f'{m:.3f}' if key == 'objective' else f'{100 * m:.0f} %',
                    ha='center', va='bottom', fontsize=7, color=INK)
        ax.set_xticks([])
        ax.set_title(title, fontsize=8.5, color=INK, loc='left')
        style(ax)
        if key != 'objective':
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{100 * v:.0f} %'))
    axes[0].set_ylim(0, 0.46)
    axes[1].set_ylim(0, 0.27)
    axes[2].set_ylim(0, 0.85)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=COLOR[p], label=LABEL[p]) for p in order], loc='upper center',
               ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.08), handlelength=1.2)
    fig.tight_layout(w_pad=1.2)
    return fig


def robustness(plt):
    rows = [r for r in load(ROOT / 'artifacts/robustness/runs.jsonl') if r['group'] == 'variant']
    panels = [
        ('Traffic load', [(v, x) for v, x in (('load-x1', 1), ('load-x1.5', 1.5), ('load-x2', 2),
                                              ('load-x2.5', 2.5), ('base', 3), ('load-x4', 4), ('load-x5', 5))
                          if any(r['variant'] == v for r in rows)], 'load (x base)', lambda v: f'x{v:g}'),
        ('Feature-extraction cost', [('extract-1us', 1.0), ('extract-3us', 3.0), ('base', 7.5)],
         'µs per packet (M2)', lambda v: f'{v:g}'),
        ('Stream-separation error', [('separation-0', 0), ('base', 2), ('separation-5', 5)],
         'misrouted flows (%)', lambda v: f'{v:g}'),
    ]
    order = ['greedy-v2', 'value-greedy-alerts', 'ps-anneal-alerts']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, (title, points, xlabel, fmt) in zip(axes, panels):
        xs = [x for _, x in points]
        for p in order:
            stats = [mean_ci([r['objective'] for r in rows if r['variant'] == v and r['policy'] == p])
                     for v, _ in points]
            ax.errorbar(xs, [m for m, _ in stats], yerr=[h for _, h in stats], color=COLOR[p],
                        marker=MARKER[p], markersize=5, linewidth=2, capsize=2.5, elinewidth=0.8,
                        markeredgecolor='white', markeredgewidth=0.8, zorder=3)
        ticks = [x for x in xs if float(x).is_integer()] if len(xs) > 4 else xs
        ax.set_xticks(ticks)
        ax.set_xticklabels([fmt(x) for x in ticks], fontsize=8, color=INK2)
        ax.set_xlabel(xlabel, fontsize=8, color=INK2)
        ax.set_title(title, fontsize=8.5, color=INK, loc='left')
        style(ax)
    axes[0].set_ylabel('objective (lower is better)', fontsize=8, color=INK2)
    axes[0].set_ylim(0, 0.55)
    # Direct labels on the last panel (identity is not colour alone).
    last = axes[2]
    for p in order:
        m, _ = mean_ci([r['objective'] for r in rows if r['variant'] == 'separation-5' and r['policy'] == p])
        last.annotate(LABEL[p], (5, m), xytext=(6, 0), textcoords='offset points', fontsize=7,
                      color=INK, va='center')
    last.set_xlim(-0.5, 9.5)
    fig.tight_layout(w_pad=1.0)
    return fig


def ablations(plt):
    rows = load(ROOT / 'artifacts/robustness/runs.jsonl')
    base = {r['seed']: r['objective'] for r in rows
            if r['group'] == 'variant' and r['variant'] == 'base' and r['policy'] == 'ps-anneal-alerts'}
    labels = ['ps-anneal (no alerts)', 'budget 10 ms', 'budget 25 ms', 'no pruning', 'no repair',
              'alert window 10 s', 'alert window 20 s', 'budget 100 ms', 'oracle values']
    names = {'ps-anneal (no alerts)': 'no alert feedback', 'oracle values': 'oracle task values'}
    stats = []
    for label in labels:
        d = [r['objective'] - base[r['seed']] for r in rows if r['group'] == 'ablation' and r['label'] == label]
        m, h = mean_ci(d)
        stats.append((names.get(label, label), 100 * m, 100 * h))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    y = list(range(len(stats)))[::-1]
    for yy, (name, m, h) in zip(y, stats):
        color = COLOR['ps-anneal'] if name != 'oracle task values' else COLOR['ps-anneal-alerts']
        ax.barh(yy, m, height=0.6, color=color, edgecolor='white', linewidth=1.5, zorder=2)
        ax.errorbar(m, yy, xerr=h, color=INK, capsize=2, linewidth=0.8, zorder=3)
        text = f'{m:+.1f}' if abs(m) >= 0.05 else '0.0'
        ax.text(m + (h + 0.3 if m >= 0 else -h - 0.3), yy, text, fontsize=7, color=INK,
                va='center', ha='left' if m >= 0 else 'right')
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in stats], fontsize=7.5, color=INK2)
    ax.set_xlabel('change in objective vs full method (points)', fontsize=7.5, color=INK2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.yaxis.grid(False)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=7.5)
    ax.set_xlim(-5, 11)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'figure.facecolor': 'white'})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in (('main-ifogsim', main_results), ('robustness', robustness), ('ablations', ablations)):
        fig = make(plt)
        for ext in ('pdf', 'png'):
            fig.savefig(OUT / f'{name}.{ext}', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(OUT / f'{name}.pdf')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
