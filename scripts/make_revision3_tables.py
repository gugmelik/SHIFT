"""LaTeX tables for revision 3 (paper_v5.tex) from the revision-2 run outputs.

Reads paper_results_revision2/{stats,summary,blocks}.json (scripts/paper_stats.py,
collect_paper_results.py, block_analysis.py) and writes article_tex/tables_v5/*.tex:

  ci.tex       per-reference means with 95 % cluster-bootstrap CIs (test set)
  effects.tex  ours (alpha=6) minus each comparator: mean difference, rank-biserial r, Holm p
  fid.tex      FID / KID against the teacher set (auxiliary)
  blocks_stats.tex  depth vs angle rank correlations + per-reference sums of single-block shares
  ablation_effects.tex  ablations: mean difference to the main configuration, r, Holm p

No numbers are typed by hand: rerun after a new GPU run.
    python scripts/make_revision3_tables.py --res paper_results_revision2 --out article_tex/tables_v5
"""
import argparse, itertools, json, os

REFS = [("monet_water_lilies", "Моне"), ("vangogh_self_portrait", "Ван~Гог"),
        ("ciurlionis_sonata_sun", "Чюрлёнис"), ("pirosmani_giraffe", "Пиросмани")]
METHODS = [("t2i", "T2I"), ("ours", "Наш, $\\alpha=6$"), ("ours_rule", "Наш, по правилу"),
           ("teacher", "Учитель (I2I)"), ("shift_text", "SHIFT, $\\alpha=6$"), ("act", "Linear-AcT")]
OTHERS = [("t2i", "T2I"), ("teacher", "учитель"), ("shift_text", "SHIFT"), ("act", "Linear-AcT")]
GRAM = "gram_to_reference"
TV = "!{\\thinvline}"


def mc(n, name, last):
    return "\\multicolumn{%d}{c%s}{%s}" % (n, "|" if last else TV, name)


def num(x, d=3, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    s = s.replace(".", "{,}")
    return s.replace("-", "$-$").replace("+", "$+$") if sign else s.replace("-", "$-$")


def pval(p):
    if p >= 0.995:
        return "1"
    if p >= 0.001:
        return num(p, 3)
    m, e = f"{p:.0e}".split("e")
    return f"${m}\\cdot 10^{{{int(e)}}}$"


def scale(metric):
    return 1e5 if metric == GRAM else 1.0


def ci_table(stats):
    cols = [("csd_to_reference", "CSD $\\uparrow$"), (GRAM, "Грам $\\downarrow$"),
            ("dinov2_to_origin", "DINOv2-исх. $\\uparrow$"), ("lpips_to_origin", "LPIPS $\\downarrow$")]
    out = ["\\begin{tabular}{|>{\\raggedright}m{30mm}" + "!{\\thinvline}>{\\centering}m{37mm}" * 3
           + "!{\\thinvline}>{\\centering\\arraybackslash}m{37mm}|}", "\\hline",
           "Метод & " + " & ".join(c[1] for c in cols) + "\\\\", "\\hline"]
    for ref, rname in REFS:
        out.append(f"\\multicolumn{{5}}{{|l|}}{{\\textit{{{rname}}}}}\\\\\\thinhline")
        for m, mname in METHODS:
            cells = []
            for metric, _ in cols:
                v = stats["test"][ref][m].get(metric)
                if not v or v.get("mean") is None:
                    cells.append("\\---"); continue
                k = scale(metric); lo, hi = v["ci95"]
                cells.append(f"{num(v['mean']*k)} [{num(lo*k)}; {num(hi*k)}]")
            out.append(f"{mname} & " + " & ".join(cells) + "\\\\\\thinhline")
    out[-1] = out[-1].replace("\\\\\\thinhline", "\\\\")
    out += ["\\hline", "\\end{tabular}"]
    return "\n".join(out)


def effects_table(stats):
    mets = [(GRAM, "Грам"), ("csd_to_reference", "CSD"), ("dinov2_to_origin", "DINOv2-исх.")]
    head = " & ".join(mc(3, n, i == 2) for i, (_, n) in enumerate(mets))
    out = ["\\begin{tabular}{|>{\\raggedright}m{31mm}" + "!{\\thinvline}>{\\centering}m{14mm}!{\\thinvline}>{\\centering}m{9mm}!{\\thinvline}>{\\centering}m{18mm}" * 2
           + "!{\\thinvline}>{\\centering}m{14mm}!{\\thinvline}>{\\centering}m{9mm}!{\\thinvline}>{\\centering\\arraybackslash}m{18mm}|}",
           "\\hline", "Сравнение & " + head + "\\\\",
           " & " + " & ".join(["$\\Delta$ & $r$ & $p_{\\mathrm{Holm}}$"] * 3) + "\\\\", "\\hline"]
    for ref, rname in REFS:
        out.append(f"\\multicolumn{{10}}{{|l|}}{{\\textit{{{rname}}}}}\\\\\\thinhline")
        tests = stats["tests"][ref]
        for o, oname in OTHERS:
            cells = []
            for metric, _ in mets:
                t = next((x for x in tests if x["other"] == o and x["metric"] == metric), None)
                if t is None:
                    cells += ["\\---"] * 3; continue
                cells += [num(t["mean_diff"] * scale(metric), 3, sign=True), num(t["r_rb"], 2, sign=True), pval(t["p_holm"])]
            out.append(f"наш $-$ {oname} & " + " & ".join(cells) + "\\\\\\thinhline")
    out[-1] = out[-1].replace("\\\\\\thinhline", "\\\\")
    out += ["\\hline", "\\end{tabular}"]
    return "\n".join(out)


def fid_table(summary):
    ms = [("t2i", "T2I"), ("shift_text", "SHIFT, $\\alpha=6$"), ("ours", "Наш, $\\alpha=6$"), ("ours_rule", "Наш, по правилу")]
    out = ["\\begin{tabular}{|>{\\raggedright}m{30mm}" + "!{\\thinvline}>{\\centering}m{11mm}" * 7
           + "!{\\thinvline}>{\\centering\\arraybackslash}m{11mm}|}", "\\hline",
           " & " + " & ".join(mc(2, r, i == 3) for i, (_, r) in enumerate(REFS)) + "\\\\",
           "Метод & " + " & ".join(["FID & KID"] * 4) + "\\\\", "\\hline"]
    for m, mname in ms:
        cells = []
        for ref, _ in REFS:
            t = summary["references"][ref]["test"][m]
            cells += [num(t["fid"]["mean"], 1), num(t["kid_mean"]["mean"] * 1e3, 2)]
        out.append(f"{mname} & " + " & ".join(cells) + "\\\\\\thinhline")
    out[-1] = out[-1].replace("\\\\\\thinhline", "\\\\")
    out += ["\\hline", "\\end{tabular}"]
    return "\n".join(out)


def _ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i]); r = [0] * len(x)
    for k, i in enumerate(order): r[i] = k + 1
    return r


def spearman(x, y):
    rx, ry, n = _ranks(x), _ranks(y), len(x)
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / (n * (n * n - 1))


def perm_p(x, y):  # exact two-sided permutation p (8! permutations)
    r0, hit, tot = abs(spearman(x, y)), 0, 0
    for perm in itertools.permutations(y):
        tot += 1; hit += abs(spearman(x, list(perm))) >= r0 - 1e-12
    return hit / tot


def block_stats(blocks):
    g, p = blocks["geometry"], blocks["pooled"]
    ks = [str(k) for k in range(1, 9)]
    ang = [g["angle_deg"][k] for k in ks]; depth = [-int(k) for k in ks]
    rel = [-g["vector_to_token_norm"][k] for k in ks]
    res = {"rho_angle_depth": spearman(ang, depth), "metrics": {}}
    for m in ("csd_to_reference", GRAM, "lpips_to_origin", "dinov3_to_reference"):
        fo = [p["frac_only"][m][k] for k in ks]
        res["metrics"][m] = {"angle": (spearman(fo, ang), perm_p(fo, ang)),
                             "depth": (spearman(fo, depth), perm_p(fo, depth)),
                             "neg_v_over_h": (spearman(fo, rel), perm_p(fo, rel))}
    res["additivity"] = {r: blocks["per_ref"][r]["additivity"] for r, _ in REFS}
    res["additivity_pooled"] = p["additivity"]
    res["full_effect"] = {r: blocks["per_ref"][r]["effects"]["full"] for r, _ in REFS}
    names = [("csd_to_reference", "CSD"), (GRAM, "$\\ln$ Грам"), ("lpips_to_origin", "LPIPS"), ("dinov3_to_reference", "DINOv3-реф.")]
    out = ["\\begin{tabular}{|>{\\raggedright}m{20mm}" + (TV + ">{\\centering}m{23mm}") * 3
           + (TV + ">{\\centering}m{10mm}") * 4 + TV + ">{\\centering\\arraybackslash}m{10mm}|}",
           "\\hline",
           "Метрика & $\\rho$ с углом & $\\rho$ с глубиной & $\\rho$ с $-\\|v\\|/\\|h\\|$ & \\multicolumn{5}{c|}{Сумма вкладов отдельных блоков}\\\\",
           " & & & & Моне & Ван~Гог & Чюрл. & Пиросм. & среднее\\\\", "\\hline"]
    for m, n in names:
        c = res["metrics"][m]
        f = lambda t: f"{num(t[0], 2)} ($p={pval(t[1]).strip('$')}$)" if t[1] >= 0.001 else f"{num(t[0], 2)} ($p<0{{,}}001$)"
        add = [num(res["additivity"][r][m], 2) for r, _ in REFS] + [num(res["additivity_pooled"][m], 2)]
        out.append(f"{n} & {f(c['angle'])} & {f(c['depth'])} & {f(c['neg_v_over_h'])} & " + " & ".join(add) + "\\\\\\thinhline")
    out[-1] = out[-1].replace("\\\\\\thinhline", "\\\\")
    out += ["\\hline", "\\end{tabular}"]
    return "\n".join(out), res


def ablation_table(stats):
    cfgs = [("blocks_0-3", "Только блоки 1--4"), ("blocks_4-7", "Только блоки 5--8"), ("steps_0", "Только шаг 1"),
            ("steps_0-1", "Шаги 1--2"), ("steps_2-3", "Шаги 3--4"), ("pooled", "Вектор, усреднённый по позициям"),
            ("with_txt", "С текстовым потоком"), ("svm", "С классификатором SVM")]
    mets = [("csd_to_reference", "CSD"), (GRAM, "Грам"), ("dinov2_to_origin", "DINOv2-исх.")]
    head = " & ".join(mc(3, n, i == 2) for i, (_, n) in enumerate(mets))
    out = ["\\begin{tabular}{|>{\\raggedright}m{36mm}" + "!{\\thinvline}>{\\centering}m{12mm}!{\\thinvline}>{\\centering}m{9mm}!{\\thinvline}>{\\centering}m{14mm}" * 2
           + "!{\\thinvline}>{\\centering}m{12mm}!{\\thinvline}>{\\centering}m{9mm}!{\\thinvline}>{\\centering\\arraybackslash}m{14mm}|}",
           "\\hline", "Конфигурация & " + head + "\\\\",
           " & " + " & ".join(["$\\Delta$ & $r$ & $p_{\\mathrm{Holm}}$"] * 3) + "\\\\", "\\hline"]
    for c, cname in cfgs:
        cells = []
        for metric, _ in mets:
            t = next((x for x in stats["ablation_pooled"] if x["config"] == c and x["metric"] == metric), None)
            cells += ([num(t["mean_diff"] * scale(metric), 3, sign=True), num(t["r_rb"], 2, sign=True), pval(t["p_holm"])]
                      if t else ["\\---"] * 3)
        out.append(f"{cname} & " + " & ".join(cells) + "\\\\\\thinhline")
    out[-1] = out[-1].replace("\\\\\\thinhline", "\\\\")
    out += ["\\hline", "\\end{tabular}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="paper_results_revision2")
    ap.add_argument("--out", default="article_tex/tables_v5")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stats = json.load(open(os.path.join(a.res, "stats.json"), encoding="utf-8"))
    summary = json.load(open(os.path.join(a.res, "summary.json"), encoding="utf-8"))
    blocks = json.load(open(os.path.join(a.res, "blocks.json"), encoding="utf-8"))
    files = {"ci.tex": ci_table(stats), "effects.tex": effects_table(stats), "fid.tex": fid_table(summary),
             "ablation_effects.tex": ablation_table(stats)}
    files["blocks_stats.tex"], res = block_stats(blocks)
    for n, t in files.items():
        open(os.path.join(a.out, n), "w", encoding="utf-8").write(t + "\n")
    json.dump(res, open(os.path.join(a.out, "blocks_stats.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
