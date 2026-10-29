"""Evaluate review prioritization using Cake disagreement as a *surrogate* target."""
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent.parent
PROJECT=Path(r'C:\Users\Administrator\WorkBuddy AI\年龄评估')
rows=json.loads((PROJECT/'reports/pred_all.json').read_text(encoding='utf-8'))
a=np.array([r['anno'] for r in rows]);p=np.array([r['pred'] for r in rows]);c=np.array([r['cake'] for r in rows])
t=np.array([r['ts'] for r in rows]);o=np.array([r['ori'] for r in rows])
d=a!=c
out={}

def rank_auc(target,score):
    order=np.argsort(score);n=len(score);ranks=np.zeros(n);i=0
    while i<n:
        j=i+1
        while j<n and score[order[j]]==score[order[i]]:j+=1
        ranks[order[i:j]]=(i+1+j)/2;i=j
    n1=target.sum();n0=n-n1
    return float((ranks[target].sum()-n1*(n1+1)/2)/(n1*n0))

for key,score in [('model_vs_cake',abs(p-c)),('ts_vs_cake',abs(t-c)),('ori_vs_cake',abs(o-c))]:
    z={'disagreement_auc':rank_auc(d,score),'base_rate':float(d.mean()),'thresholds':{},'budgets':{}}
    for threshold in [2,5,8,10]:
        flag=score>=threshold
        tp=int(np.sum(flag&d));n=int(flag.sum())
        z['thresholds'][str(threshold)]={'flagged':n,'hits':tp,'precision':tp/n if n else None,'recall':tp/int(d.sum())}
    order=np.argsort(-score,kind='stable')
    for fraction in [.05,.1,.2]:
        k=int(np.ceil(len(rows)*fraction));tp=int(d[order[:k]].sum())
        z['budgets'][str(fraction)]={'k':k,'hits':tp,'precision':tp/k,'recall':tp/int(d.sum()),
                                    'cutoff':float(score[order[k-1]]),'ties_at_cutoff':int(np.sum(score==score[order[k-1]]))}
    out[key]=z
minimum=np.minimum.reduce([abs(p-c),abs(p-t),abs(p-o)])
flags=minimum>=8
early=(p>=15)&(p<=25)
out['actual_verdict_far_from_all_candidates']={'qualifying':int(flags.sum()),'human_cake_disagreements':int(np.sum(flags&d)),
          'suppressed_by_prior_15_25_branch':int(np.sum(flags&early)),
          'shown_far_candidate_warning':int(np.sum(flags&~early))}
out['warning_coverage']={'true_ge20':int(np.sum(a>=20)),'true_ge20_outside_pred_15_25':int(np.sum((a>=20)&~early)),
        'true_15_25_outside_pred_15_25':int(np.sum((a>=15)&(a<=25)&~early)),
        'true_15_bias':float((p-a)[a==15].mean()),'true_16_bias':float((p-a)[a==16].mean())}
out['near_boundary_ranking']={f'{lo}-{hi}':rank_auc((a[(a>=lo)&(a<=hi)]>=20),p[(a>=lo)&(a<=hi)]) for lo,hi in [(10,29),(15,25),(18,21)]}
out['candidate_constant_checks']={str(k):{'closer_anno_count':int(np.sum(abs(k-a[d])<abs(k-c[d]))),
       'tie_count':int(np.sum(abs(k-a[d])==abs(k-c[d]))),'mae_on_disagreements':float(np.mean(abs(k-a[d])))} for k in [14,14.5,15]}
out['model_cake_total_mae_difference']=float(np.mean(abs(p-a)-abs(c-a)))
out['disagreement_improvement_gap_gt5_share']=float(np.sum((abs(c-a)-abs(p-a))[(abs(c-a)>5)&d])/np.sum((abs(c-a)-abs(p-a))[d]))
out['kept_cake_rows']={'n':int((~d).sum()),'model_mae':float(abs(p-a)[~d].mean()),'cake_mae':0}
out['explanatory_two_distance_count']={'closer_anno_but_not_more_accurate_than_cake':int(np.sum(d&(abs(p-a)<abs(p-c))&(abs(p-a)>=abs(c-a)))),
                                    'more_accurate_than_cake_but_closer_to_cake':int(np.sum(d&(abs(p-a)<abs(c-a))&(abs(p-a)>abs(p-c))))}
(BASE/'outputs'/'audit_review_value.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
