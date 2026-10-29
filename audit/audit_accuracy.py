"""Recompute age accuracy from per-record predictions without changing source data."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def ratio(a,b):
    return a/b if b else None


def metrics(rows, threshold):
    if not rows:
        return {'n':0}
    errors=[r['pred']-r['anno'] for r in rows]
    absolute=sorted(abs(e) for e in errors)
    pos=.9*(len(absolute)-1)
    lo=int(pos);hi=math.ceil(pos)
    tp=sum(r['anno']>=threshold and r['pred']>=threshold for r in rows)
    tn=sum(r['anno']<threshold and r['pred']<threshold for r in rows)
    fp=sum(r['anno']<threshold and r['pred']>=threshold for r in rows)
    fn=len(rows)-tp-tn-fp
    recall=ratio(tp,tp+fn);specificity=ratio(tn,tn+fp)
    return dict(n=len(rows),mae=statistics.mean(absolute),
        median_absolute_error=statistics.median(absolute),
        p90_absolute_error=absolute[lo]+(absolute[hi]-absolute[lo])*(pos-lo),
        rmse=math.sqrt(statistics.mean(e*e for e in errors)),bias=statistics.mean(errors),
        within={str(t):{'count':sum(e<=t for e in absolute),'fraction':sum(e<=t for e in absolute)/len(rows)} for t in [1,2,3,5,10]},
        boundary=dict(threshold=threshold,tp=tp,tn=tn,fp=fp,fn=fn,
            recall_ge_threshold=recall,specificity_lt_threshold=specificity,
            precision_ge_threshold=ratio(tp,tp+fp),accuracy=(tp+tn)/len(rows),
            balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None,
            all_below_threshold_baseline=sum(r['anno']<threshold for r in rows)/len(rows)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions',type=Path)
    parser.add_argument('--output',type=Path,default=Path('audit-output/metrics.json'))
    parser.add_argument('--threshold',type=float,default=20)
    args=parser.parse_args()
    if not math.isfinite(args.threshold):parser.error('threshold must be finite')
    if args.output.resolve()==args.predictions.resolve():parser.error('output must differ from input')
    blob=args.predictions.read_bytes()
    raw=json.loads(blob.decode('utf-8-sig'))
    if not isinstance(raw,list) or not raw:parser.error('input must be a non-empty JSON array')
    rows=[]
    for i,r in enumerate(raw):
        if not isinstance(r,dict):parser.error(f'row {i}: expected object')
        row={}
        for key in ['anno','pred','cake']:
            if key=='cake' and r.get(key) is None:continue
            value=r.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                parser.error(f'row {i}: {key} must be a finite number')
            row[key]=float(value)
        rows.append(row)
    result={'input_sha256':hashlib.sha256(blob).hexdigest(),
        'reference':'anno; prediction is compared as a floating point value without UI rounding',
        'overall':metrics(rows,args.threshold),
        'age_bands':{f'{lo}-{hi}':metrics([r for r in rows if lo<=r['anno']<hi+1],args.threshold)
            for lo,hi in [(0,6),(7,12),(13,15),(16,19),(20,29),(30,39),(40,59),(60,100)]}}
    cake=[r for r in rows if 'cake' in r]
    if cake:
        result['cake_comparison']={
            'n':len(cake),
            'model_closer_to_anno_than_to_cake':sum(abs(r['pred']-r['anno'])<abs(r['pred']-r['cake']) for r in cake),
            'model_error_less_than_cake_error':sum(abs(r['pred']-r['anno'])<abs(r['cake']-r['anno']) for r in cake),
            'model_mae':statistics.mean(abs(r['pred']-r['anno']) for r in cake),
            'cake_mae':statistics.mean(abs(r['cake']-r['anno']) for r in cake)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result['overall'],ensure_ascii=False,indent=2))
    print(f'Saved {args.output}')


if __name__=='__main__':main()
