"""Independent, read-only audit. Writes aggregate results only under this chat's outputs/."""
import collections
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PROJECT = Path(r"C:\Users\Administrator\WorkBuddy AI\年龄评估")
DATA = Path(r"F:\yll\10.21标年龄")
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"
SRC = PROJECT / "reports/pred_all.json"
rows = json.loads(SRC.read_text(encoding="utf-8"))
a = np.array([r["anno"] for r in rows], float)
p = np.array([r["pred"] for r in rows], float)
c = np.array([r["cake"] for r in rows], float)
err = p - a
ae = abs(err)
N = len(a)
result = {"archive_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(), "n": N}


def wilson(k, n):
    if not n:
        return None
    z = statistics.NormalDist().inv_cdf(.975)
    q = k / n
    d = 1 + z*z/n
    mid = (q + z*z/(2*n))/d
    half = z * math.sqrt(q*(1-q)/n + z*z/(4*n*n))/d
    return [mid-half, mid+half]


def metric(mask, pred=p):
    x, y = a[mask], pred[mask]
    e = y - x
    ab = abs(e)
    if not len(e):
        return {"n": 0}
    true = x >= 20
    guess = y >= 20
    tp = int(np.sum(true & guess))
    tn = int(np.sum(~true & ~guess))
    fp = int(np.sum(~true & guess))
    fn = int(np.sum(true & ~guess))
    tpr = tp/(tp+fn) if tp+fn else None
    tnr = tn/(tn+fp) if tn+fp else None
    return dict(n=len(e), mae=float(ab.mean()), median=float(np.median(ab)),
                p90_linear=float(np.quantile(ab,.9)), bias=float(e.mean()),
                rmse=float(np.sqrt(np.mean(e*e))),
                within={str(t): dict(count=int(np.sum(ab <= t)), fraction=float(np.mean(ab <= t)),
                                     wilson95=wilson(int(np.sum(ab <= t)),len(e)))
                        for t in (1,2,3,5,10)},
                tp=tp,tn=tn,fp=fp,fn=fn,tpr=tpr,tnr=tnr,
                precision=tp/(tp+fp) if tp+fp else None,
                npv=tn/(tn+fn) if tn+fn else None,
                accuracy=(tp+tn)/len(e),baseline_accuracy=float(np.mean(~true)),
                balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,
                tpr_wilson95=wilson(tp,tp+fn),tnr_wilson95=wilson(tn,tn+fp))


allmask = np.ones(N,bool)
result["overall"] = metric(allmask)
result["overall"]["p90_script_orderstat"] = float(np.sort(ae)[int(.9*(N-1))])
result["bands"] = {f"{lo}-{hi}":metric((a>=lo)&(a<=hi)) for lo,hi in
                   [(0,6),(7,12),(13,15),(16,19),(20,29),(30,39)]}
result["per_age"] = {str(int(age)):metric(a==age) for age in np.unique(a)}
result["windows"] = {f"{lo}-{hi}":metric((a>=lo)&(a<=hi)) for lo,hi in
                     [(10,29),(13,27),(15,25),(18,21),(16,23)]}
folders = np.array([Path(r["path"]).parent.name for r in rows])
genders = np.array([r.get("anno_gender") for r in rows])
result["folders"] = {s:metric(folders==s) for s in sorted(set(folders),key=int)}
result["genders"] = {str(s):metric(genders==s) for s in set(genders)}
result["macro_mae_six_bands"] = float(np.mean([v["mae"] for v in result["bands"].values()]))
result["macro_mae_individual_ages"] = float(np.mean([v["mae"] for v in result["per_age"].values()]))
result["equal_under20_over20_mae"] = float((ae[a<20].mean()+ae[a>=20].mean())/2)
result["constant_baselines"] = {str(k):metric(allmask,np.full(N,k)) for k in [14.5,15,statistics.median(a)]}
result["candidate_baselines"] = {key:metric(allmask,np.array([r[key] for r in rows],float))
                                  for key in ["cake","ori","ts"]}

# The report's predicate compares two distances measured FROM THE MODEL.
different = a.astype(int) != c.astype(int)
gap = abs(a-c)
near_anno = abs(p-a) < abs(p-c)
near_cake = abs(p-a) > abs(p-c)
better_than_cake = abs(p-a) < abs(c-a)


def comparison(mask):
    n = int(mask.sum())
    if not n:
        return {"n":0}
    return dict(n=n,closer_anno=int(np.sum(mask&near_anno)),closer_cake=int(np.sum(mask&near_cake)),
                closer_anno_fraction=float(near_anno[mask].mean()),
                model_actually_less_error_than_cake=int(np.sum(mask&better_than_cake)),
                model_mae=float(ae[mask].mean()),cake_mae=float(gap[mask].mean()),
                mae_gain=float(np.mean(gap[mask]-ae[mask])),within5=int(np.sum(mask&(ae<=5))),
                anno_lower=int(np.sum(mask&(a<c))),anno_higher=int(np.sum(mask&(a>c))),
                avg_anno=float(np.mean(a[mask])),avg_cake=float(np.mean(c[mask])),avg_pred=float(np.mean(p[mask])),
                wilson95=wilson(int(np.sum(mask&near_anno)),n))


result["cake_disagreements"] = comparison(different)
result["cake_disagreements"]["gap_quantiles"] = dict(zip(["min","p25","median","p75","p90","p95","p99","max"],
                 map(float,np.quantile(gap[different],[0,.25,.5,.75,.9,.95,.99,1]))))
result["cake_gap_bins"] = {f"{lo}-{hi}":comparison(different&(gap>=lo)&(gap<=hi))
                            for lo,hi in [(1,1),(2,2),(3,5),(6,10),(11,20),(21,1000)]}
result["cake_directions"] = {"anno_lower":comparison(different&(a<c)),"anno_higher":comparison(different&(a>c))}
result["cake_gap_trim"] = {f"gap_le_{t}":comparison(different&(gap<=t)) for t in [2,5,10,20]}
result["cake_simple_choice_lower_fraction"] = float(np.mean(a[different]<c[different]))
result["cake_constant_closeness"] = {str(k):float(np.mean(abs(k-a[different])<abs(k-c[different])))
                                     for k in [10,14.5,15,statistics.median(a)]}
result["cake_other_candidate_closeness"] = {key:float(np.mean(abs(np.array([r[key] for r in rows])[different]-a[different]) <
                                           abs(np.array([r[key] for r in rows])[different]-c[different]))) for key in ["ori","ts"]}
idx = np.flatnonzero(different)
gains = gap[idx]-ae[idx]
order = np.argsort(gap[idx])[::-1]
result["cake_extreme_contribution"] = {str(k):dict(gap_sum_fraction=float(gap[idx[order[:k]]].sum()/gap[idx].sum()),
                     gain_sum_fraction=float(gains[order[:k]].sum()/gains.sum()),
                     remaining=comparison(different & ~np.isin(np.arange(N),idx[order[:k]]))) for k in [5,10,25,50]}

# Compare with the actual candidate chooser, including exact and age-error outcomes.
choices = []
for r in rows:
    candidates = [r[k] for k in ("ori","ts","cake") if r[k] is not None]
    choices.append(min(candidates,key=lambda v:abs(v-r["pred"])))
choices = np.array(choices,float)
result["nearest_candidate"] = {"overall":metric(allmask,choices),"disagreements":metric(different,choices),
                "exact_on_disagreements":int(np.sum(different&(choices==a))),
                "anno_present_on_disagreements":int(np.sum(different&np.array([r["anno"] in [r[k] for k in ("ori","ts","cake")] for r in rows])))}

# Rounding is intentionally measured separately from floating-point accuracy.
color = lambda e: 0 if e<=2 else 1 if e<=5 else 2 if e<=10 else 3
result["colors"] = {key:dict(zip(["green","yellow","orange","red"],
                          [int(np.sum(np.array([color(x) for x in values])==k)) for k in range(4)]))
                    for key,values in [("float",ae),("rounded",abs(np.round(err)))]}
result["colors"]["different_rows"] = sum(color(x)!=color(y) for x,y in zip(ae,abs(np.round(err))))
probs = np.array([r["probs"] for r in rows],float)
labels = rows[0]["labels"]
mid_code = np.array([1.5,6.5,15,25,35,45,55,65,85])
result["archive_integrity"] = dict(unique_paths=len(set(r["path"].lower() for r in rows)),
      finite=bool(np.isfinite(p).all() and np.isfinite(a).all() and np.isfinite(probs).all()),
      anno_noninteger=int(np.sum(a!=np.round(a))),
      labels_consistent=all(r["labels"]==labels for r in rows),
      max_probability_sum_deviation=float(abs(probs.sum(axis=1)-1).max()),
      max_pred_vs_probability_dot_deviation=float(abs(p-probs@mid_code).max()),
      labels=labels)


def ranking(score):
    positive=a>=20
    order=np.argsort(score)
    ranks=np.zeros(N)
    j=0
    while j<N:
        end=j+1
        while end<N and score[order[end]]==score[order[j]]:
            end+=1
        ranks[order[j:end]]=(j+1+end)/2
        j=end
    n1=int(positive.sum()); n0=N-n1
    auc=(ranks[positive].sum()-n1*(n1+1)/2)/(n1*n0)
    order=np.argsort(-score)
    yy=positive[order]
    ss=score[order]
    ends=np.flatnonzero(np.r_[ss[:-1]!=ss[1:],True])
    cum=np.cumsum(yy)[ends]
    ap=np.sum((cum/((ends+1)))*(np.diff(np.r_[0,cum])/n1))
    return {"roc_auc":float(auc),"average_precision":float(ap),"positive_prevalence":n1/N}


result["ranking"] = {"scalar_age":ranking(p),"sum_probability_ge20":ranking(probs[:,3:].sum(axis=1))}
result["classification_alternatives"] = {}
for name,yes in {"twenty_gt_teen":probs[:,3]>probs[:,2],"all_nine_argmax_ge20":np.argmax(probs,axis=1)>=3}.items():
    result["classification_alternatives"][name]=metric(allmask,np.where(yes,20.,19.))

# Descriptive confidence intervals: IID faces first, then entire-folder resampling.
rng=np.random.default_rng(20261022)
boot={"mae":[],"within2":[],"within5":[],"accuracy_minus_majority":[],"balanced_accuracy":[],"cake_closeness":[]}
true=a>=20; predicted=p>=20
correct=(true==predicted).astype(float)
majority=(~true).astype(float)
pos=np.flatnonzero(true); neg=np.flatnonzero(~true); sub=np.flatnonzero(different)
for _ in range(50):
    samples=rng.integers(0,N,(100,N))
    boot["mae"].extend(ae[samples].mean(axis=1))
    boot["within2"].extend((ae[samples]<=2).mean(axis=1))
    boot["within5"].extend((ae[samples]<=5).mean(axis=1))
    boot["accuracy_minus_majority"].extend((correct-majority)[samples].mean(axis=1))
    s1=pos[rng.integers(0,len(pos),(100,len(pos)))];s0=neg[rng.integers(0,len(neg),(100,len(neg)))]
    boot["balanced_accuracy"].extend((predicted[s1].mean(axis=1)+(~predicted[s0]).mean(axis=1))/2)
    sb=sub[rng.integers(0,len(sub),(100,len(sub)))]
    boot["cake_closeness"].extend(near_anno[sb].mean(axis=1))
result["bootstrap_iid_95"]={k:list(map(float,np.quantile(v,[.025,.975]))) for k,v in boot.items()}
folder_names=sorted(set(folders),key=int)
sizes=np.array([(folders==s).sum() for s in folder_names]); sums=np.array([ae[folders==s].sum() for s in folder_names])
draws=rng.integers(0,len(sizes),(10000,len(sizes)))
result["bootstrap_folder_mae_95"]=list(map(float,np.quantile(sums[draws].sum(axis=1)/sizes[draws].sum(axis=1),[.025,.975])))
result["leave_one_folder_out_mae"]={s:float(ae[folders!=s].mean()) for s in folder_names}
print("Independent metrics calculated",flush=True)

# Audit raw source coverage and metadata. No production-data write API is used.
inventory={}; source={}; hashes=collections.defaultdict(list); manifest={}
for folder in map(str,range(7,14)):
    d=DATA/folder
    jpgs=sorted(x for x in d.rglob("*") if x.is_file() and x.suffix.lower() in (".jpg",".jpeg"))
    inv=collections.Counter(images=len(jpgs))
    inv["json_files"]=sum(1 for _ in d.rglob("*.json"))
    for img in jpgs:
        blob=img.read_bytes()
        sha=hashlib.sha256(blob).hexdigest()
        hashes[sha].append(str(img))
        st=img.stat();manifest[str(img)]=[st.st_size,st.st_mtime_ns,sha]
        mode=orientation=None
        try:
            with Image.open(img) as im:
                mode=im.mode;orientation=im.getexif().get(274,1); dims=im.size
                im.load()
            inv['mode_'+str(mode)]+=1
            inv['exif_'+str(orientation)]+=1
        except Exception as ex:
            inv['decode_failures']+=1
            dims=None
        jp=Path(str(img)+'.json')
        if not jp.exists():
            inv['missing_json']+=1
            continue
        jb=jp.read_bytes();st=jp.stat();manifest[str(jp)]=[st.st_size,st.st_mtime_ns,hashlib.sha256(jb).hexdigest()]
        try:
            ob=json.loads(jb.decode('utf-8'))
        except Exception:
            inv['json_decode_failures']+=1
            continue
        faces=ob.get('result') or []
        inv['faces']+=len(faces)
        inv['multi_face_images']+=int(len(faces)>1)
        for face in faces:
            anno=face.get('anno_age')
            inv['labeled_faces']+=int(anno is not None)
            if anno is None:
                continue
            source[str(img).lower()]=face
            b=face.get('box') or {}
            if not b:
                inv['labeled_without_box']+=1
                continue
            bb=[float(b[x]) for x in ['left','top','width','height']]
            inv['labeled_fractional_box']+=int(any(x!=int(x) for x in bb))
            inv['labeled_invalid_box']+=int(bb[2]<=0 or bb[3]<=0)
            if dims:
                w,h=dims
                inv['box_outside_raw_image']+=int(bb[0]<0 or bb[1]<0 or bb[0]+bb[2]>w or bb[1]+bb[3]>h)
    inventory[folder]=dict(inv)
    print('Read-only source scan folder',folder,dict(inv),flush=True)
result['source_inventory']=inventory
result['source_total']={k:sum(v.get(k,0) for v in inventory.values()) for k in set().union(*(v.keys() for v in inventory.values()))}
archive_paths={r['path'].lower() for r in rows}
result['source_archive_comparison']={'missing_labeled_in_archive':len(set(source)-archive_paths),
      'archive_without_current_label':len(archive_paths-set(source)),
      'field_mismatches':{}}
for archive_key,raw_key in [('anno','anno_age'),('anno_gender','anno_gender'),('cake','cake_age'),('ori','ori_age'),('ts','ts_age')]:
    result['source_archive_comparison']['field_mismatches'][archive_key]=sum(
         r[archive_key]!=source.get(r['path'].lower(),{}).get(raw_key) for r in rows)
dups=[v for v in hashes.values() if len(v)>1]
result['duplicate_images']={'unique_hashes':len(hashes),'groups':len(dups),'redundant_images':sum(len(v)-1 for v in dups),
       'archive_redundant_images':sum(max(0,sum(x.lower() in archive_paths for x in v)-1) for v in dups),
       'cross_folder_groups':sum(len(set(Path(x).parent.name for x in v))>1 for v in dups)}
stat_changes=[]
for path,(size,mtime,sha) in manifest.items():
    st=Path(path).stat()
    if (st.st_size,st.st_mtime_ns)!=(size,mtime):
        stat_changes.append(path)
result['source_metadata_changed_during_scan']=len(stat_changes)
(BASE/'work'/'source_read_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8')
OUT.mkdir(exist_ok=True)
(OUT/'audit_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print('Saved aggregate audit_metrics.json',flush=True)
