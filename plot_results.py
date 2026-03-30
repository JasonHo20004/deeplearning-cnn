#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize training results from all models.
Generates an interactive HTML dashboard with accuracy and loss charts.

Usage:
    python plot_results.py

Output: checkpoints/training_dashboard.html
"""

import os
import re
import json


def parse_summary_log(filepath):
    """Parse summary log files (BTImages*.txt, CIFAR10 .txt, M1 log .txt).
    Format: 'DD/MM/YYYY HH:MM:SS Epoch X/Y summary: loss_train=..., acc_train=...%, ...'
    """
    epochs, loss_train, acc_train, acc_val = [], [], [], []
    loss_val = []

    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Extract epoch number
            m_epoch = re.search(r'Epoch\s+(\d+)/(\d+)', line)
            if not m_epoch:
                continue
            ep = int(m_epoch.group(1))

            # Extract loss_train
            m_lt = re.search(r'loss_train=([\d.]+)', line)
            lt = float(m_lt.group(1)) if m_lt else None

            # Extract acc_train
            m_at = re.search(r'acc_train=([\d.]+)%', line)
            at = float(m_at.group(1)) if m_at else None

            # Extract loss_val
            m_lv = re.search(r'loss_val=([\d.]+)', line)
            lv = float(m_lv.group(1)) if m_lv else None

            # Extract acc_val
            m_av = re.search(r'acc_val=([\d.]+)%', line)
            av = float(m_av.group(1)) if m_av else None

            epochs.append(ep)
            loss_train.append(lt)
            acc_train.append(at)
            loss_val.append(lv)
            acc_val.append(av)

    if not epochs:
        return None

    return {
        'epochs': epochs,
        'loss_train': loss_train,
        'acc_train': acc_train,
        'loss_val': loss_val,
        'acc_val': acc_val,
    }


def parse_data_plot_files(checkpoint_dir):
    """Parse separate data files: Loss_plot.txt, train_prec1.txt, val_prec1.txt.
    Format: 'epoch_number value' or 'epoch_number tensor(value)'
    """
    def read_values(filepath):
        vals = {}
        if not os.path.exists(filepath):
            return vals
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    ep = int(parts[0])
                    # Handle tensor(...) format
                    val_str = parts[1]
                    m = re.search(r'tensor\(([\d.]+)\)', val_str)
                    if m:
                        val = float(m.group(1))
                    else:
                        try:
                            val = float(val_str)
                        except ValueError:
                            continue
                    vals[ep] = val
        return vals

    loss = read_values(os.path.join(checkpoint_dir, 'Loss_plot.txt'))
    train_prec1 = read_values(os.path.join(checkpoint_dir, 'train_prec1.txt'))
    val_prec1 = read_values(os.path.join(checkpoint_dir, 'val_prec1.txt'))

    if not loss and not train_prec1:
        return None

    all_epochs = sorted(set(list(loss.keys()) + list(train_prec1.keys()) + list(val_prec1.keys())))

    return {
        'epochs': all_epochs,
        'loss_train': [loss.get(e) for e in all_epochs],
        'acc_train': [train_prec1.get(e) for e in all_epochs],
        'loss_val': [None for _ in all_epochs],  # not stored separately
        'acc_val': [val_prec1.get(e) for e in all_epochs],
    }


def parse_m1_csv(filepath):
    """Parse M1 CSV log files.
    Format: epoch,lr,loss_train,acc_train(%),loss_val,acc_val(%)
    """
    if not os.path.exists(filepath):
        return None

    epochs, loss_train, acc_train, loss_val, acc_val = [], [], [], [], []

    with open(filepath, 'r', encoding='utf-8') as f:
        import csv
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None
        for row in reader:
            if len(row) < 6:
                continue
            try:
                epochs.append(int(row[0]))
                loss_train.append(float(row[2]))
                acc_train.append(float(row[3]))
                loss_val.append(float(row[4]))
                acc_val.append(float(row[5]))
            except (ValueError, IndexError):
                continue

    if not epochs:
        return None

    return {
        'epochs': epochs,
        'loss_train': loss_train,
        'acc_train': acc_train,
        'loss_val': loss_val,
        'acc_val': acc_val,
    }


def collect_all_data():
    """Collect training data from all checkpoint directories."""
    models = {}

    # --- NetBT3 (BTImages 224x224) ---
    bt3_dir = './checkpoints/BTImages5_NetBT3_'
    bt3_summary = parse_summary_log(os.path.join(bt3_dir, 'BTImages5_NetBT3.txt'))
    bt3_plots = parse_data_plot_files(bt3_dir)
    # Prefer plot files (they have more accurate loss), merge with summary for acc_val loss
    if bt3_plots:
        if bt3_summary:
            bt3_plots['loss_val'] = bt3_summary['loss_val']
        models['NetBT3 (224×224)'] = bt3_plots
    elif bt3_summary:
        models['NetBT3 (224×224)'] = bt3_summary

    # --- NetBT2 (BTImages 224x224) ---
    bt2_dir = './checkpoints/BTImages5_NetBT2_'
    bt2_summary = parse_summary_log(os.path.join(bt2_dir, 'BTImages5_NetBT2.txt'))
    bt2_plots = parse_data_plot_files(bt2_dir)
    if bt2_plots:
        if bt2_summary:
            bt2_plots['loss_val'] = bt2_summary['loss_val']
        models['NetBT2 (224×224)'] = bt2_plots
    elif bt2_summary:
        models['NetBT2 (224×224)'] = bt2_summary

    # --- CIFAR10 ---
    cifar_log = './checkpoints/CIFAR10_MDFNet/.txt'
    cifar_data = parse_summary_log(cifar_log)
    if cifar_data:
        models['CIFAR10_MDFNet (32×32)'] = cifar_data

    # --- M1 224x224 ---
    m1_224_csv = './checkpoints/M1_224/training_log_224.csv'
    m1_224_data = parse_m1_csv(m1_224_csv)
    if m1_224_data:
        models['M1 (224×224)'] = m1_224_data
    else:
        m1_224_txt = './checkpoints/M1_224/training_log_224.txt'
        m1_224_data = parse_summary_log(m1_224_txt)
        if m1_224_data:
            models['M1 (224×224)'] = m1_224_data

    # --- M1 32x32 ---
    m1_32_csv = './checkpoints/M1_32/training_log_32.csv'
    m1_32_data = parse_m1_csv(m1_32_csv)
    if m1_32_data:
        models['M1 (32×32)'] = m1_32_data
    else:
        m1_32_txt = './checkpoints/M1_32/training_log_32.txt'
        m1_32_data = parse_summary_log(m1_32_txt)
        if m1_32_data:
            models['M1 (32×32)'] = m1_32_data

    return models


def generate_html(models, output_path):
    """Generate an interactive HTML dashboard using Chart.js."""

    # Prepare data for JS
    chart_data = {}
    for name, data in models.items():
        chart_data[name] = {
            'epochs': data['epochs'],
            'loss_train': [v if v is not None else None for v in data['loss_train']],
            'acc_train': [v if v is not None else None for v in data['acc_train']],
            'loss_val': [v if v is not None else None for v in data['loss_val']],
            'acc_val': [v if v is not None else None for v in data['acc_val']],
        }

    # Color palette
    colors = [
        {'main': '#6366f1', 'light': 'rgba(99,102,241,0.15)'},   # indigo
        {'main': '#f43f5e', 'light': 'rgba(244,63,94,0.15)'},    # rose
        {'main': '#10b981', 'light': 'rgba(16,185,129,0.15)'},   # emerald
        {'main': '#f59e0b', 'light': 'rgba(245,158,11,0.15)'},   # amber
        {'main': '#8b5cf6', 'light': 'rgba(139,92,246,0.15)'},   # violet
        {'main': '#06b6d4', 'light': 'rgba(6,182,212,0.15)'},    # cyan
    ]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Dashboard — CNN Models</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
            color: #e2e8f0;
            min-height: 100vh;
            padding: 2rem;
        }}
        .header {{
            text-align: center;
            margin-bottom: 2.5rem;
        }}
        .header h1 {{
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        .header p {{
            color: #94a3b8;
            font-size: 0.95rem;
        }}
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: rgba(30, 30, 60, 0.7);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 12px;
            padding: 1.2rem;
            backdrop-filter: blur(10px);
            transition: transform 0.2s, border-color 0.2s;
        }}
        .stat-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.5);
        }}
        .stat-card .model-name {{
            font-size: 0.8rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.3rem;
        }}
        .stat-card .best-acc {{
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }}
        .stat-card .detail {{
            font-size: 0.8rem;
            color: #64748b;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        @media (max-width: 900px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
        }}
        .chart-container {{
            background: rgba(30, 30, 60, 0.7);
            border: 1px solid rgba(99, 102, 241, 0.15);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
        }}
        .chart-container h2 {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #c7d2fe;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .chart-container h2 .icon {{
            font-size: 1.2rem;
        }}
        .chart-wrapper {{
            position: relative;
            height: 350px;
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .full-width .chart-wrapper {{
            height: 400px;
        }}
        .legend-custom {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            justify-content: center;
            margin-top: 1rem;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.85rem;
            color: #94a3b8;
            cursor: pointer;
            transition: opacity 0.2s;
        }}
        .legend-item:hover {{
            opacity: 0.8;
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        .no-data {{
            text-align: center;
            color: #64748b;
            padding: 4rem 2rem;
            font-size: 1.1rem;
        }}
        .no-data .emoji {{
            font-size: 3rem;
            margin-bottom: 1rem;
        }}
        .table-container {{
            background: rgba(30, 30, 60, 0.7);
            border: 1px solid rgba(99, 102, 241, 0.15);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            overflow-x: auto;
        }}
        .table-container h2 {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #c7d2fe;
            margin-bottom: 1rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        th {{
            text-align: left;
            padding: 0.7rem 1rem;
            border-bottom: 2px solid rgba(99, 102, 241, 0.3);
            color: #a5b4fc;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}
        td {{
            padding: 0.6rem 1rem;
            border-bottom: 1px solid rgba(99, 102, 241, 0.1);
            color: #cbd5e1;
        }}
        tr:hover td {{
            background: rgba(99, 102, 241, 0.05);
        }}
        .badge {{
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
        }}
        .badge-best {{
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
        }}
        footer {{
            text-align: center;
            margin-top: 2rem;
            color: #475569;
            font-size: 0.8rem;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>📊 Training Dashboard</h1>
            <p>CNN Model Training Results — Accuracy & Loss Visualization</p>
        </div>
        <div id="stats-container" class="stats-grid"></div>
        <div id="charts-container" class="charts-grid"></div>
        <div id="table-container" class="table-container" style="margin-top:1.5rem;"></div>
        <footer>
            <p>Generated by plot_results.py · deeplearning-cnn project</p>
        </footer>
    </div>

    <script>
    const DATA = {json.dumps(chart_data, indent=2)};
    const COLORS = {json.dumps(colors)};

    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = 'rgba(99,102,241,0.1)';
    Chart.defaults.font.family = "'Inter', sans-serif";

    function getColor(idx) {{
        return COLORS[idx % COLORS.length];
    }}

    const modelNames = Object.keys(DATA);

    // ==============================
    // Stats cards
    // ==============================
    const statsContainer = document.getElementById('stats-container');
    modelNames.forEach((name, idx) => {{
        const d = DATA[name];
        const validAcc = d.acc_val.filter(v => v !== null);
        const bestAcc = validAcc.length > 0 ? Math.max(...validAcc) : null;
        const bestEpoch = bestAcc !== null ? d.epochs[d.acc_val.indexOf(bestAcc)] : '—';
        const totalEpochs = d.epochs.length;
        const color = getColor(idx);

        const card = document.createElement('div');
        card.className = 'stat-card';
        card.innerHTML = `
            <div class="model-name">${{name}}</div>
            <div class="best-acc" style="color:${{color.main}}">${{bestAcc !== null ? bestAcc.toFixed(2) + '%' : 'N/A'}}</div>
            <div class="detail">Best Val Acc · Epoch ${{bestEpoch}} / ${{totalEpochs}}</div>
        `;
        statsContainer.appendChild(card);
    }});

    if (modelNames.length === 0) {{
        const noData = document.createElement('div');
        noData.className = 'no-data';
        noData.innerHTML = '<div class="emoji">📭</div><p>No training logs found.<br>Run training first, then re-generate this dashboard.</p>';
        document.getElementById('charts-container').appendChild(noData);
    }}

    // ==============================
    // Chart helper
    // ==============================
    function createChart(containerId, title, icon, dataKey, yLabel, fullWidth) {{
        const datasets = [];
        modelNames.forEach((name, idx) => {{
            const d = DATA[name];
            const vals = d[dataKey];
            if (!vals || vals.every(v => v === null || v === 0)) return;
            const color = getColor(idx);
            datasets.push({{
                label: name,
                data: d.epochs.map((ep, i) => ({{ x: ep, y: vals[i] }})).filter(p => p.y !== null),
                borderColor: color.main,
                backgroundColor: color.light,
                borderWidth: 2,
                pointRadius: d.epochs.length > 50 ? 0 : 2,
                pointHoverRadius: 5,
                tension: 0.3,
                fill: false,
            }});
        }});

        if (datasets.length === 0) return;

        const wrapper = document.createElement('div');
        wrapper.className = 'chart-container' + (fullWidth ? ' full-width' : '');
        wrapper.innerHTML = `
            <h2><span class="icon">${{icon}}</span> ${{title}}</h2>
            <div class="chart-wrapper"><canvas id="${{containerId}}"></canvas></div>
        `;
        document.getElementById('charts-container').appendChild(wrapper);

        new Chart(document.getElementById(containerId), {{
            type: 'line',
            data: {{ datasets }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false,
                }},
                plugins: {{
                    legend: {{
                        position: 'bottom',
                        labels: {{
                            usePointStyle: true,
                            pointStyle: 'circle',
                            padding: 15,
                            font: {{ size: 12 }},
                        }},
                    }},
                    tooltip: {{
                        backgroundColor: 'rgba(15, 15, 35, 0.95)',
                        titleColor: '#e2e8f0',
                        bodyColor: '#94a3b8',
                        borderColor: 'rgba(99,102,241,0.3)',
                        borderWidth: 1,
                        cornerRadius: 8,
                        padding: 12,
                        callbacks: {{
                            label: function(ctx) {{
                                const suffix = yLabel.includes('%') ? '%' : '';
                                return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(2) + suffix;
                            }}
                        }}
                    }},
                }},
                scales: {{
                    x: {{
                        type: 'linear',
                        title: {{ display: true, text: 'Epoch', color: '#64748b' }},
                        grid: {{ color: 'rgba(99,102,241,0.06)' }},
                    }},
                    y: {{
                        title: {{ display: true, text: yLabel, color: '#64748b' }},
                        grid: {{ color: 'rgba(99,102,241,0.06)' }},
                    }},
                }},
            }},
        }});
    }}

    // Create 4 charts
    createChart('chart-acc-train', 'Training Accuracy', '🎯', 'acc_train', 'Accuracy (%)', false);
    createChart('chart-acc-val', 'Validation Accuracy', '✅', 'acc_val', 'Accuracy (%)', false);
    createChart('chart-loss-train', 'Training Loss', '📉', 'loss_train', 'Loss', false);
    createChart('chart-loss-val', 'Validation Loss', '📈', 'loss_val', 'Loss', false);

    // ==============================
    // Summary table
    // ==============================
    const tableContainer = document.getElementById('table-container');
    if (modelNames.length > 0) {{
        let rows = '';
        modelNames.forEach((name, idx) => {{
            const d = DATA[name];
            const validAccTrain = d.acc_train.filter(v => v !== null);
            const validAccVal = d.acc_val.filter(v => v !== null);
            const validLossTrain = d.loss_train.filter(v => v !== null);
            const validLossVal = d.loss_val.filter(v => v !== null && v !== 0);

            const bestTrainAcc = validAccTrain.length > 0 ? Math.max(...validAccTrain).toFixed(2) : '—';
            const bestValAcc = validAccVal.length > 0 ? Math.max(...validAccVal).toFixed(2) : '—';
            const finalLoss = validLossTrain.length > 0 ? validLossTrain[validLossTrain.length - 1].toFixed(4) : '—';
            const finalValLoss = validLossVal.length > 0 ? validLossVal[validLossVal.length - 1].toFixed(4) : '—';
            const totalEpochs = d.epochs.length;

            const bestValIdx = validAccVal.length > 0 ? d.acc_val.indexOf(Math.max(...validAccVal)) : -1;
            const bestEpoch = bestValIdx >= 0 ? d.epochs[bestValIdx] : '—';

            const color = getColor(idx);
            rows += `<tr>
                <td><span style="color:${{color.main}};font-weight:600">●</span> ${{name}}</td>
                <td>${{totalEpochs}}</td>
                <td>${{bestTrainAcc}}%</td>
                <td><strong>${{bestValAcc}}%</strong> <span class="badge badge-best">epoch ${{bestEpoch}}</span></td>
                <td>${{finalLoss}}</td>
                <td>${{finalValLoss}}</td>
            </tr>`;
        }});

        tableContainer.innerHTML = `
            <h2>📋 Summary Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Epochs</th>
                        <th>Best Train Acc</th>
                        <th>Best Val Acc</th>
                        <th>Final Train Loss</th>
                        <th>Final Val Loss</th>
                    </tr>
                </thead>
                <tbody>${{rows}}</tbody>
            </table>
        `;
    }}
    </script>
</body>
</html>""";

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Dashboard saved to: {output_path}")
    return output_path


def main():
    print("=" * 50)
    print("  Collecting training logs...")
    print("=" * 50)

    models = collect_all_data()

    if not models:
        print("\nNo training logs found!")
        print("Expected log locations:")
        print("  - checkpoints/BTImages5_NetBT3_/")
        print("  - checkpoints/BTImages5_NetBT2_/")
        print("  - checkpoints/CIFAR10_MDFNet/")
        print("  - checkpoints/M1_224/")
        print("  - checkpoints/M1_32/")
        return

    print(f"\nFound {len(models)} model(s):")
    for name, data in models.items():
        n_epochs = len(data['epochs'])
        valid_acc = [v for v in data['acc_val'] if v is not None]
        best = max(valid_acc) if valid_acc else 0
        print(f"  • {name}: {n_epochs} epochs, best val acc = {best:.2f}%")

    output_path = './checkpoints/training_dashboard.html'
    generate_html(models, output_path)
    print(f"\n✅ Open this file in your browser to view the charts:")
    print(f"   {os.path.abspath(output_path)}")


if __name__ == '__main__':
    main()
