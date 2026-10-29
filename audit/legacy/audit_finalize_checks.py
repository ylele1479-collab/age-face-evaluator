"""Final read-only provenance checks and targeted verification of a batch-size hypothesis."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path
import numpy as np

os.environ['HF_HUB_OFFLINE']='1'
PROJECT=Path(r'C:\Users\Administrator\WorkBuddy AI\年龄评估')
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT))
rr=json.loads((PROJECT/'reports/pred_all.json').read_text(encoding='utf-8'))
f=BASE/'outputs'/'audit_inference.json'
info=json.loads(f.read_text(encoding='utf-8'))
if 'current_2500' in info:
    info['supplemental_snapshot']=info.pop('current_2500')
info['snapshot_note']='The 2411 archive is fixed; 116 additional labeled records were observed in a later read-only scan. The 2527-record union is a supplemental snapshot, not a claim about the current live total.'
f.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
manifest=json.loads((BASE/'work'/'source_read_manifest.json').read_text(encoding='utf-8'))
changed=[];missing=0
for path,(size,mtime,sha) in manifest.items():
    p=Path(path)
    if not p.exists():missing+=1;continue
    nowsha=hashlib.sha256(p.read_bytes()).hexdigest()
    if nowsha!=sha:changed.append(path)
archive_mismatches={key:0 for key in ['anno','anno_gender','cake','ori','ts']}
for row in rr:
    raw=json.loads(Path(row['path']+'.json').read_text(encoding='utf-8'))['result'][0]
    for k,s in [('anno','anno_age'),('anno_gender','anno_gender'),('cake','cake_age'),('ori','ori_age'),('ts','ts_age')]:
        archive_mismatches[k]+=int(row[k]!=raw.get(s))
checks={'archive_sha256':hashlib.sha256((PROJECT/'reports/pred_all.json').read_bytes()).hexdigest(),
      'source_manifest_file_count':len(manifest),'source_content_changes_since_first_scan':len(changed),
      'changed_images':sum(not p.endswith('.json') for p in changed),'changed_jsons':sum(p.endswith('.json') for p in changed),
      'source_missing_files':missing,'archive_vs_live_field_mismatches':archive_mismatches}

# Control batch size while leaving model, weights, processor and crop fixed.
from age_face_tool.core.pipeline import load_backend,MODEL_VIT
from age_face_tool.core.imageio import imread_bgr
backend=load_backend(MODEL_VIT)
random.seed(0)
sample=random.sample(rr,50)
batch_tests=[]
for row in sample[:5]:
    raw=json.loads(Path(row['path']+'.json').read_text(encoding='utf-8'))['result'][0]
    box=tuple(float(raw['box'][k]) for k in ['left','top','width','height'])
    img=imread_bgr(row['path'])
    pp=[x for _,x in backend.prepare_views(img,box)]
    a2,_,_=backend.predict_probs(pp)
    a4,_,_=backend.predict_probs(pp+pp)
    batch_tests.append({'batch2_archive_delta':float(np.mean(a2)-row['pred']),
                        'batch4_archive_delta':float(np.mean(a4[:2])-row['pred']),
                        'batch_size_only_delta':float(np.mean(a4[:2])-np.mean(a2))})
checks['controlled_batch_size_tests']=batch_tests

# Paired uncertainty for a direct candidate comparison.
a=np.array([r['anno'] for r in rr]);p=np.array([r['pred'] for r in rr]);t=np.array([r['ts'] for r in rr])
d=abs(p-a)-abs(t-a);rng=np.random.default_rng(20261022);boot=[]
for _ in range(50):
    ix=rng.integers(0,len(rr),(100,len(rr)));boot.extend(d[ix].mean(axis=1))
checks['model_minus_ts_mae']={'point':float(d.mean()),'paired_iid_bootstrap95':list(map(float,np.quantile(boot,[.025,.975])))}
(BASE/'outputs'/'audit_provenance_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
