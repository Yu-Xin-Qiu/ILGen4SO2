"""Global versus local generation around the measured highest-AC IL, with 4D Pareto screening."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from final_pair_evaluation import cached_pair_sa, internal_diversity
from models import create_model
from molecular_tokenizer import MolecularTokenizer
from text_oracles import TextOracle
from train_pair_generator import (
    canonical_pair, generate_pairs, pair_length_statistics, representation,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/IECR_section33_50k_20261023'
SEEDS = [11, 23, 37]
SEED_CATION = 'CCCCCC[n+]1cccc(C)c1'
SEED_ANION = 'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F'
SIGMA = 0.25
Z = 1.645
COLORS = {'global': '#356CB1', 'local': '#DD8452'}


def set_cell(cell, text, bold=False, size=7.0):
    p = cell.paragraphs[0]
    p.text = ''
    run = p.add_run(str(text))
    run.bold = bold
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)


def make_table(doc, headers, rows, widths, size=7.0):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for j, (cell, width) in enumerate(zip(table.rows[0].cells, widths)):
        set_cell(cell, headers[j], bold=True, size=size)
        cell.width = Inches(width)
    for values in rows:
        row = table.add_row()
        for j, value in enumerate(values):
            set_cell(row.cells[j], value, bold=False, size=size)
            row.cells[j].width = Inches(widths[j])
    borders = OxmlElement('w:tblBorders')
    for side in ['top', 'bottom', 'left', 'right', 'insideH', 'insideV']:
        element = OxmlElement('w:' + side)
        element.set(qn('w:val'), 'single')
        element.set(qn('w:sz'), '4')
        element.set(qn('w:color'), 'D9D9D9')
        borders.append(element)
    table._tbl.tblPr.append(borders)
    for row in table.rows:
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.line_spacing = 1


@torch.no_grad()
def generate_with_latent(model, tokenizer, language, max_length, latents, device,
                         generation_seed=20261023, condition_vector=None):
    model.eval()
    records = []
    for start in range(0, len(latents), 256):
        z = torch.as_tensor(latents[start:start + 256], device=device, dtype=torch.float32)
        size = z.size(0)
        prefix = torch.tensor([[tokenizer.bos_id, tokenizer.vocab['<CAT>']]] * size, device=device)
        conditions = (torch.zeros(size, 3, device=device) if condition_vector is None
                      else torch.as_tensor(condition_vector, device=device).expand(size, -1))
        ids = prefix
        for _ in range(max_length - ids.size(1)):
            logits = model.decode_logits(ids, z, conditions)[:, -1]
            values, indices = logits.topk(min(32, logits.size(-1)), dim=-1)
            next_token = indices.gather(1, torch.multinomial(torch.softmax(values, -1), 1))
            ids = torch.cat([ids, next_token], dim=1)
            if ids[:, -1].eq(tokenizer.eos_id).all():
                break
        for sequence in ids.cpu().tolist():
            try:
                cation, anion, cation_repr, anion_repr = tokenizer.decode_pair(sequence)
                valid = canonical_pair(cation, anion)
                if valid is None:
                    raise ValueError('invalid charge or component count')
                cation, anion, pair = valid
                error = ''
            except Exception as exc:
                cation = anion = pair = cation_repr = anion_repr = ''
                error = type(exc).__name__
            records.append({
                'cation_smiles': cation, 'anion_smiles': anion, 'IL_SMILES': pair,
                'cation_representation': cation_repr, 'anion_representation': anion_repr,
                'valid_pair': not bool(error), 'error': error,
            })
    return pd.DataFrame(records)


def pareto_ranks(frame):
    objectives = frame[['AC_lcb', 'DE_lcb', 'S_lcb', 'SA_ucb']].to_numpy(float)
    objectives[:, :3] *= -1.0
    try:
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        _, ranks = NonDominatedSorting().do(objectives, return_rank=True)
        return ranks
    except Exception:
        ranks = np.zeros(len(frame), dtype=int)
        for index, point in enumerate(objectives):
            dominates = np.all(objectives <= point, axis=1) & np.any(objectives < point, axis=1)
            ranks[index] = int(dominates.sum())
        return ranks


def score_arm(generated, oracle, known_pairs, known_ions, seed, arm, output_dir):
    valid = generated[generated.valid_pair].drop_duplicates('IL_SMILES').copy()
    valid = valid[valid.IL_SMILES.map(lambda value: canonical_pair(*value.split('.', 1)) is not None)]
    if len(valid) == 0:
        raise RuntimeError(f'No valid pairs for {arm} seed {seed}')
    valid['SA_score'] = cached_pair_sa(valid)
    valid['pair_novel'] = ~valid.IL_SMILES.isin(known_pairs)
    valid['both_ions_novel'] = valid.apply(
        lambda row: row.cation_smiles not in known_ions and row.anion_smiles not in known_ions, axis=1)
    cat_known = valid.cation_smiles.isin(known_ions)
    an_known = valid.anion_smiles.isin(known_ions)
    known_pair = valid.IL_SMILES.isin(known_pairs)
    valid['known_pair'] = known_pair
    valid['known_ion_recombination'] = cat_known & an_known & ~known_pair
    valid['new_ion_pair'] = ~(cat_known & an_known)
    scores = oracle.score(valid.IL_SMILES.tolist(), physical_folds=[6, 7, 8, 9, 10], gas_folds=[4, 5])
    valid = pd.concat([valid.reset_index(drop=True), scores.drop(columns=['IL_SMILES'])], axis=1)
    valid['AC'] = valid.x_SO2_298
    valid['DE'] = valid.x_SO2_298 - valid.x_SO2_313
    valid['S'] = valid.x_SO2_298 / (valid.x_CO2_298 + 1e-12)
    valid['AC_lcb'] = valid.AC - Z * valid.AC * valid.ln_x_SO2_298_sd
    valid['DE_lcb'] = valid.DE - Z * np.sqrt(
        (valid.AC * valid.ln_x_SO2_298_sd) ** 2
        + (valid.x_SO2_313 * valid.ln_x_SO2_313_sd) ** 2
    )
    valid['S_lcb'] = valid.S - Z * valid.S * np.sqrt(
        valid.ln_x_SO2_298_sd ** 2 + valid.ln_x_CO2_298_sd ** 2
    )
    valid['SA_ucb'] = valid.SA_score
    valid['hard_pass_uq'] = (
        (valid.Tm_K + Z * valid.Tm_K_sd < 298.15)
        & (valid.eta_298_mPas * np.exp(Z * valid.ln_eta_298_sd) < 100)
        & (valid.toxicity - Z * valid.toxicity_sd > 3.7)
    )
    valid['physical_gas'] = (
        valid[['x_SO2_298', 'x_SO2_313', 'x_CO2_298']].gt(0).all(axis=1)
        & valid[['x_SO2_298', 'x_SO2_313', 'x_CO2_298']].le(1).all(axis=1)
    )
    physical = valid[valid.physical_gas & valid.SA_score.notna()].copy()
    physical['pareto_rank'] = pareto_ranks(physical)
    valid['pareto_rank'] = np.nan
    valid.loc[physical.index, 'pareto_rank'] = physical['pareto_rank']
    valid['source'] = arm
    valid['seed'] = seed
    valid.to_parquet(output_dir / f'{arm}_{seed}_scored.parquet', index=False)
    metrics = {
        'arm': arm, 'seed': seed, 'generated': int(len(generated)),
        'valid_pair_rate': float(generated.valid_pair.mean()),
        'unique_valid_pair_rate': float(len(valid) / max(len(generated), 1)),
        'HCSR': float(valid.hard_pass.sum() / max(len(generated), 1)),
        'HCSR_UQ': float(valid.hard_pass_uq.sum() / max(len(generated), 1)),
        'AC_mean': float(valid.AC.mean()), 'DE_mean': float(valid.DE.mean()),
        'S_mean': float(valid.S.mean()), 'SA_mean': float(valid.SA_score.mean()),
        'pareto_count': int((valid.pareto_rank == 0).sum()),
        'pareto_rate': float((valid.pareto_rank == 0).sum() / max(len(generated), 1)),
        'both_ions_novelty': float(valid.both_ions_novel.mean()),
        'known_pair_rate': float(valid.known_pair.mean()),
        'internal_diversity': internal_diversity(valid.IL_SMILES.tolist(), seed),
    }
    return valid, metrics


def main():
    while not (OUT / 'SECTION33_COMPLETED.json').exists():
        print('waiting for Section 3.3...', flush=True)
        import time
        time.sleep(60)
    config = yaml.safe_load((OUT / 'config.yaml').read_text(encoding='utf-8'))
    known = pd.read_csv(OUT / 'data/known_pairs.csv')
    known_pairs = set(known.IL_SMILES)
    known_ions = set(known.cation_smiles) | set(known.anion_smiles)
    language, model_name = 'selfies', 'vae'
    tokenizer = MolecularTokenizer.load(OUT / 'pair_pool' / f'vocab_{language}_pair.json')
    pool = pd.read_parquet(OUT / 'pair_pool' / 'pair_pool_scored.parquet')
    maximum, keep, priors_all = pair_length_statistics(pool, tokenizer, language)
    priors = {key: values[keep] for key, values in priors_all.items()}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    oracle = TextOracle(device=device)
    output_dir = OUT / 'global_local'
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, sweep_rows = [], []
    for seed in SEEDS:
        model = create_model(model_name, len(tokenizer.vocab), maximum, tokenizer,
                             config['model'], condition_dim=3).to(device)
        model.load_state_dict(torch.load(OUT / 'iteration_runs' / str(seed) / 'final_model.pt',
                                         map_location=device, weights_only=True))
        cat_repr = representation(SEED_CATION, language)
        ani_repr = representation(SEED_ANION, language)
        seed_ids = torch.tensor([tokenizer.encode_pair(cat_repr, ani_repr, maximum)], device=device)
        conditions = torch.zeros(1, 3, device=device)
        with torch.no_grad():
            mu, _ = model.encode(seed_ids, conditions)
        z_seed = mu[0].cpu().numpy()
        np.save(output_dir / f'z_seed_{seed}.npy', z_seed)

        global_generated = generate_pairs(model, tokenizer, language, maximum, 1000, device, priors,
                                          True, generation_seed=seed + 20_000)
        global_generated.to_parquet(output_dir / f'global_{seed}_generated.parquet', index=False)
        _, global_metrics = score_arm(global_generated, oracle, known_pairs, known_ions, seed,
                                      'global', output_dir)
        rows.append(global_metrics)
        print(json.dumps(global_metrics), flush=True)

        rng = np.random.default_rng(seed + 30_000)
        z_local = z_seed[None, :] + SIGMA * rng.normal(size=(1000, z_seed.size))
        local_generated = generate_with_latent(model, tokenizer, language, maximum, z_local, device,
                                               generation_seed=seed + 30_000)
        local_generated.to_parquet(output_dir / f'local_{seed}_generated.parquet', index=False)
        _, local_metrics = score_arm(local_generated, oracle, known_pairs, known_ions, seed,
                                     'local', output_dir)
        rows.append(local_metrics)
        print(json.dumps(local_metrics), flush=True)

        for sigma in [0.10, 0.50]:
            rng = np.random.default_rng(seed + int(sigma * 1e6))
            z_sigma = z_seed[None, :] + sigma * rng.normal(size=(1000, z_seed.size))
            generated = generate_with_latent(model, tokenizer, language, maximum, z_sigma, device,
                                             generation_seed=seed + int(sigma * 1e6))
            _, metrics = score_arm(generated, oracle, known_pairs, known_ions, seed,
                                   f'local_sigma{sigma:.2f}', output_dir)
            sweep_rows.append({'sigma': sigma, **metrics})
            print(json.dumps(sweep_rows[-1]), flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / 'tables/section34_global_local_raw.csv', index=False)
    summary_rows = []
    for arm, group in raw.groupby('arm'):
        for metric in ['valid_pair_rate', 'unique_valid_pair_rate', 'HCSR', 'HCSR_UQ',
                       'AC_mean', 'DE_mean', 'S_mean', 'SA_mean', 'pareto_count',
                       'pareto_rate', 'both_ions_novelty', 'internal_diversity']:
            values = group[metric].astype(float)
            summary_rows.append({'arm': arm, 'metric': metric, 'mean': values.mean(),
                                 'sd': values.std(ddof=1), 'n': len(values)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / 'tables/section34_global_local_summary.csv', index=False)
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(OUT / 'tables/section34_sigma_sweep.csv', index=False)
    table = summary.set_index(['arm', 'metric'])

    all_scored = pd.concat([
        pd.read_parquet(path) for path in sorted(output_dir.glob('global_*_scored.parquet'))
    ] + [
        pd.read_parquet(path) for path in sorted(output_dir.glob('local_[0-9]*_scored.parquet'))
    ], ignore_index=True)
    all_scored.to_parquet(output_dir / 'all_scored.parquet', index=False)

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': .8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'legend.frameon': False, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2))
    ax = axes[0, 0]
    data = [all_scored[all_scored.source == arm].AC.dropna().to_numpy() for arm in ['global', 'local']]
    bp = ax.boxplot(data, labels=['Global', 'Local'], patch_artist=True, widths=.55,
                    medianprops={'color': 'white', 'linewidth': 1})
    for patch, color in zip(bp['boxes'], [COLORS['global'], COLORS['local']]):
        patch.set_facecolor(color); patch.set_alpha(.8)
    ax.set_ylabel('AC (x_SO2, 298 K)')
    ax.text(-.16, 1.06, 'a', transform=ax.transAxes, fontweight='bold', fontsize=11)

    ax = axes[0, 1]
    data = [all_scored[all_scored.source == arm].S.dropna().to_numpy() for arm in ['global', 'local']]
    bp = ax.boxplot(data, labels=['Global', 'Local'], patch_artist=True, widths=.55,
                    medianprops={'color': 'white', 'linewidth': 1})
    for patch, color in zip(bp['boxes'], [COLORS['global'], COLORS['local']]):
        patch.set_facecolor(color); patch.set_alpha(.8)
    ax.set_ylabel('S (SO2/CO2 selectivity)')
    ax.text(-.16, 1.06, 'b', transform=ax.transAxes, fontweight='bold', fontsize=11)

    ax = axes[1, 0]
    physical = all_scored[all_scored.physical_gas & all_scored.SA_score.notna()]
    for arm in ['global', 'local']:
        subset = physical[physical.source == arm]
        ax.scatter(subset.AC_lcb, subset.S_lcb, s=7, alpha=.35, color=COLORS[arm], label=arm.capitalize())
    ax.set_xlabel('AC lower confidence bound')
    ax.set_ylabel('S lower confidence bound')
    ax.legend(loc='upper left', markerscale=2)
    ax.text(-.16, 1.06, 'c', transform=ax.transAxes, fontweight='bold', fontsize=11)

    ax = axes[1, 1]
    x = np.arange(2)
    for j, metric in enumerate(['HCSR', 'HCSR_UQ']):
        means = [table.loc[('global', metric), 'mean'], table.loc[('local', metric), 'mean']]
        sds = [table.loc[('global', metric), 'sd'], table.loc[('local', metric), 'sd']]
        ax.bar(x + (j - .5) * .32, means, .32, yerr=sds, capsize=2,
               color=[COLORS['global'], COLORS['local']][j], label=metric,
               linewidth=.4, edgecolor='white')
    ax.set_xticks(x, ['Global', 'Local'])
    ax.set_ylabel('Frozen-oracle rate')
    ymax = max(table.loc[('global', 'HCSR'), 'mean'], table.loc[('local', 'HCSR'), 'mean']) * 1.35
    ax.set_ylim(0, max(0.55, ymax))
    ax.legend(loc='upper left')
    ax.text(-.16, 1.06, 'd', transform=ax.transAxes, fontweight='bold', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.subplots_adjust(hspace=.55, wspace=.35)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / 'figures' / f'Figure_8_global_local.{ext}', dpi=400, bbox_inches='tight')
    plt.close(fig)

    def ms(arm, metric):
        row = table.loc[(arm, metric)]
        return f'{row["mean"]:.3f} ± {row["sd"]:.3f}'

    doc = Document()
    doc.styles['Normal'].font.name = 'Times New Roman'
    doc.styles['Normal'].font.size = Pt(10.5)
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = Inches(.7)
    section.left_margin = section.right_margin = Inches(.85)
    p = doc.add_paragraph()
    r = p.add_run('3.4 Global versus local generation and uncertainty-aware Pareto screening')
    r.bold = True
    r.font.size = Pt(14)
    paragraphs = [
        'The measured IL with the highest SO2 absorption at 298.15 K and 1 bar in the experimental '
        'dataset was used as the local seed: 1-hexyl-3-methylpyridinium '
        'bis(trifluoromethanesulfonyl)imide (ln x_SO2 = -0.158). The pair model from Section 3.3 was used '
        'without retraining. Global generation sampled the latent prior z ~ N(0, I); local generation used '
        'z = z_seed + 0.25 ε with ε ~ N(0, I). Each arm generated 1,000 pairs per seed for seeds 11, 23, and 37. '
        'All candidates were scored with the frozen evaluation oracle (physical folds 6–10; gas folds 4–5).',
        f'Global generation produced a valid-pair rate of {ms("global", "valid_pair_rate")} and a unique valid-pair '
        f'rate of {ms("global", "unique_valid_pair_rate")}; local generation produced '
        f'{ms("local", "valid_pair_rate")} and {ms("local", "unique_valid_pair_rate")}, respectively. '
        f'The frozen-oracle HCSR was {ms("global", "HCSR")} for global and {ms("local", "HCSR")} for local; '
        f'HCSR_UQ was {ms("global", "HCSR_UQ")} and {ms("local", "HCSR_UQ")}.',
        f'Mean AC was {ms("global", "AC_mean")} for global and {ms("local", "AC_mean")} for local; mean DE was '
        f'{ms("global", "DE_mean")} and {ms("local", "DE_mean")}; mean S was {ms("global", "S_mean")} and '
        f'{ms("local", "S_mean")}; mean SA score was {ms("global", "SA_mean")} and {ms("local", "SA_mean")}. '
        f'The number of non-dominated candidates was {ms("global", "pareto_count")} for global and '
        f'{ms("local", "pareto_count")} for local. The local arm therefore enriches the neighborhood of the '
        'measured seed, whereas the global arm explores a broader region of pair space.',
        'For the four-objective screen, AC, DE, and S were replaced by their lower confidence bounds '
        '(mean − 1.645 SD), and SA was replaced by its upper confidence bound; the Pareto rank was computed on '
        'these conservative estimates. This makes the shortlist robust to predictor uncertainty rather than '
        'ranking on point predictions alone. A sigma sweep (0.10, 0.25, 0.50) is reported in the Supporting '
        'Information; the main comparison uses sigma = 0.25.',
    ]
    for text in paragraphs:
        p = doc.add_paragraph(text)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.5
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(OUT / 'figures/Figure_8_global_local.png'), width=Inches(6.3))
    cap = doc.add_paragraph('Figure 8. Global versus local generation around the measured highest-AC IL '
                            '(3 seeds, 1,000 pairs per arm). (a) AC distribution; (b) SO2/CO2 selectivity; '
                            '(c) conservative AC–S plane; (d) frozen-oracle HCSR and HCSR_UQ. Error bars are sample SD.')
    for run in cap.runs:
        run.font.size = Pt(9)

    c = doc.add_paragraph('Table 6. Global versus local generation (3 seeds, 1,000 pairs per arm).')
    c.runs[0].bold = True
    c.runs[0].font.size = Pt(9)
    c.paragraph_format.keep_with_next = True
    rows_ = []
    for arm in ['global', 'local']:
        rows_.append([arm.capitalize(), ms(arm, 'valid_pair_rate'), ms(arm, 'unique_valid_pair_rate'),
                      ms(arm, 'HCSR'), ms(arm, 'HCSR_UQ'), ms(arm, 'AC_mean'),
                      ms(arm, 'S_mean'), ms(arm, 'SA_mean'), ms(arm, 'pareto_count')])
    make_table(doc, ['Arm', 'Valid pair', 'Unique valid', 'HCSR', 'HCSR_UQ', 'AC', 'S', 'SA', 'Pareto n'],
               rows_, [.75, .8, .85, .7, .75, .8, .8, .8, .75])
    note = doc.add_paragraph('Values are mean ± sample SD across seeds 11, 23, and 37. The local arm uses '
                             'z = z_seed + 0.25 ε. The Pareto screen uses AC/DE/S lower confidence bounds and '
                             'SA upper confidence bound at z = 1.645.')
    for run in note.runs:
        run.font.size = Pt(9)
    dest = OUT / 'IECR_Section34_GlobalLocal_50k.docx'
    doc.save(dest)
    (OUT / 'SECTION34_COMPLETED.json').write_text(json.dumps({'word': str(dest), 'sigma': SIGMA,
                                                               'seeds': SEEDS, 'samples': 1000}, indent=2))
    print(json.dumps({'word': str(dest)}))


if __name__ == '__main__':
    main()
