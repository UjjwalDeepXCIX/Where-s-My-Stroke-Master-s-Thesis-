"""
pipeline_v2 / mine_tuh_reports.py
Build a SAME-SOURCE stroke cohort from the TUH EEG Corpus (TUEG) clinician reports,
so stroke and control recordings share hospital, hardware and protocol.
(Approach follows Peterson et al., J Stroke Cerebrovasc Dis 2024;33:107714, who built
374 stroke vs 1,385 healthy TUH subjects this way.)

Usage:
  1) With your NEDC account, download only the report .txt files of TUEG, e.g.
       rsync -auxvL --include='*/' --include='*.txt' --exclude='*' \
         nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg/v2.0.1/ ./TUEG_reports/
     (check the current version folder name on the NEDC site; password is in your NEDC approval email)
  2) python mine_tuh_reports.py ./TUEG_reports  -> writes tuh_stroke_candidates.csv
  3) MANUALLY review the candidates (read the CLINICAL HISTORY / IMPRESSION lines) and keep
     confirmed ischaemic/haemorrhagic strokes with a known onset; then rsync just those sessions' .edf
     into  Research_work/Datasets/TUH_stroke/   and re-run extract_all.py + experiments_tuh.py.
"""
import os, re, sys, csv

POS = re.compile(r'\b(stroke|cva|cerebrovascular accident|infarct(?:ion)?|ischemi[ac]|ischaemi[ac]|'
                 r'(?:intra)?cerebral h(?:a)?emorrhage|ich\b|mca (?:territory|occlusion)|lacunar)', re.I)
NEG = re.compile(r'\b(no|denies|without|negative for|rule out|r/o|history of|h/o|hx of|prior|remote|old|'
                 r'family history|fh of|risk of|concern for|question of|\?|possible|rule-out)\b[^.;\n]{0,40}$', re.I)
EXCL = re.compile(r'\b(seizure|epilep|status epilepticus|tumou?r|glioma|metasta|encephalopathy|'
                  r'anoxic|hypoxic|trauma|tbi|subdural|meningitis)\b', re.I)
ACUTE = re.compile(r'\b(acute|new|recent|today|yesterday|this morning|\d+\s*(?:hours?|days?)\s*ago|'
                   r'left[- ]sided weakness|right[- ]sided weakness|hemiparesis|hemiplegia|aphasia|facial droop)\b', re.I)

def scan(root):
    rows = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if not f.endswith('.txt'): continue
            p = os.path.join(dp, f)
            try: t = open(p, errors='ignore').read()
            except Exception: continue
            hist = t[:2000]  # CLINICAL HISTORY / MEDICATIONS / INTRO usually first
            hits = [m for m in POS.finditer(hist) if not NEG.search(hist[max(0, m.start()-60):m.start()])]
            if not hits: continue
            rows.append({'report': p, 'patient': f.split('_')[0],
                         'n_pos': len(hits), 'acute_cue': bool(ACUTE.search(hist)),
                         'exclusion_terms': ';'.join(sorted({m.group(0).lower() for m in EXCL.finditer(hist)})),
                         'snippet': re.sub(r'\s+', ' ', hist[max(0, hits[0].start()-150): hits[0].end()+150])})
    return rows

if __name__ == '__main__':
    rows = scan(sys.argv[1])
    rows.sort(key=lambda r: (r['exclusion_terms'] != '', not r['acute_cue'], -r['n_pos']))
    out = os.path.join(os.path.dirname(__file__), 'out', 'tuh_stroke_candidates.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ['report']); w.writeheader(); w.writerows(rows)
    clean = [r for r in rows if r['acute_cue'] and not r['exclusion_terms']]
    print(f"{len(rows)} reports mention stroke without negation; {len(clean)} acute-cue & no exclusion terms; "
          f"{len({r['patient'] for r in clean})} unique patients -> {out}")
