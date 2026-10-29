"""Supplemental inference and code-path probes. Production files are read-only."""
import hashlib
import importlib.metadata
import json
import os
import random
import sys
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
PROJECT=Path(r'C:\Users\Administrator\WorkBuddy AI\年龄评估')
BASE=Path(__file__).resolve().parent.parent
OUT=BASE/'outputs'
sys.path.insert(0,str(PROJECT))
import numpy as np
import torch
from PIL import Image
from age_face_tool.core.imageio import imread_bgr
from age_face_tool.core.pipeline import load_backend, MODEL_VIT
from age_face_tool.core.review_data import ReviewItem
from age_face_tool.ui.review_window import ReviewWindow
from age_face_tool.core.detector import FaceBox,expand_box

rows=json.loads((PROJECT/'reports/pred_all.json').read_text(encoding='utf-8'))
data=Path(r'F:\yll\10.21标年龄')
backend=load_backend(MODEL_VIT)
info=dict(versions={x:importlib.metadata.version(x) for x in ['numpy','torch','transformers','Pillow','opencv-python']},
          commit_hash=getattr(backend.model.config,'_commit_hash',None),
          id2label=backend.model.config.id2label,midpoints=backend._ages.tolist(),
          processor=backend.processor.to_dict(),torch_num_threads=torch.get_num_threads(),
          margin=backend.margin)
print('Model loaded from local cache',info['commit_hash'],flush=True)

random.seed(0)
sample=random.sample(rows,50)
detail=[]
for i,r in enumerate(sample):
    jp=Path(r['path']+'.json')
    raw=json.loads(jp.read_text(encoding='utf-8'))['result'][0]
    box=tuple(float(raw['box'][k]) for k in ['left','top','width','height'])
    img=imread_bgr(r['path'])
    patches=[v for _,v in backend.prepare_views(img,box)]
    intpatches=[v for _,v in backend.prepare_views(img,tuple(map(int,box)))]
    patch_changed=patches[0].shape!=intpatches[0].shape or not np.array_equal(patches[0],intpatches[0])
    ages,probs,labels=backend.predict_probs(patches+intpatches if patch_changed else patches)
    pred=float(np.mean(ages[:2]))
    intpred=float(np.mean(ages[2:])) if patch_changed else pred
    detail.append(dict(archive_delta=pred-r['pred'],probability_delta=float(abs(probs[:2].mean(axis=0)-r['probs']).max()),
                       float_pred=pred,int_pred=intpred,anno=r['anno'],patch_changed=patch_changed,
                       views=list(map(float,ages[:2])),view_classification_disagree=(ages[0]>=20)!=(ages[1]>=20)))
    if (i+1)%10==0:
        print('Precision/crop/view sample',i+1,'/50',flush=True)
v=np.array([r['views'] for r in detail]);a=np.array([r['anno'] for r in detail]);p=v.mean(axis=1)
info['sample50']=dict(n=50,skipped=0,max_abs_archive_delta=max(abs(r['archive_delta']) for r in detail),
            mean_abs_archive_delta=float(np.mean([abs(r['archive_delta']) for r in detail])),
            max_probability_delta=max(r['probability_delta'] for r in detail),
            intbox_changed_crops=sum(r['patch_changed'] for r in detail),
            intbox_max_abs_pred_change=max(abs(r['int_pred']-r['float_pred']) for r in detail),
            intbox_mean_abs_pred_change=float(np.mean([abs(r['int_pred']-r['float_pred']) for r in detail])),
            intbox_mae=float(np.mean([abs(r['int_pred']-r['anno']) for r in detail])),
            floatbox_mae=float(np.mean(abs(p-a))),standard_only_mae=float(np.mean(abs(v[:,0]-a))),
            flipped_only_mae=float(np.mean(abs(v[:,1]-a))),mean_abs_view_gap=float(np.mean(abs(v[:,0]-v[:,1]))),
            max_abs_view_gap=float(np.max(abs(v[:,0]-v[:,1]))),
            view_threshold_disagreements=sum(r['view_classification_disagree'] for r in detail))

archive={r['path'].lower() for r in rows}
new=[]
for folder in range(7,14):
    for jp in sorted((data/str(folder)).glob('*.jpg.json')):
        img=jp.with_suffix('')
        if str(img).lower() in archive:
            continue
        obj=json.loads(jp.read_text(encoding='utf-8'))
        for f in obj.get('result',[]):
            if f.get('anno_age') is not None:
                new.append((img,f))
newrows=[]
for i,(path,f) in enumerate(new):
    box=tuple(float(f['box'][k]) for k in ['left','top','width','height'])
    img=imread_bgr(path)
    ages,probs,labels=backend.predict_probs([v for _,v in backend.prepare_views(img,box)])
    newrows.append(dict(path=str(path),anno=f['anno_age'],anno_gender=f.get('anno_gender'),
                        cake=f.get('cake_age'),ori=f.get('ori_age'),ts=f.get('ts_age'),
                        pred=float(np.mean(ages)),probs=list(map(float,probs.mean(axis=0))),labels=labels))
    if (i+1)%20==0:
        print('New labeled records inferred',i+1,'/',len(new),flush=True)

def metrics(rr):
    a=np.array([r['anno'] for r in rr]);p=np.array([r['pred'] for r in rr]);e=abs(p-a)
    t=a>=20;y=p>=20
    tp=int(np.sum(t&y));tn=int(np.sum(~t&~y));fp=int(np.sum(~t&y));fn=int(np.sum(t&~y))
    return dict(n=len(rr),mae=float(e.mean()),median=float(np.median(e)),bias=float((p-a).mean()),
                within2=int(np.sum(e<=2)),within5=int(np.sum(e<=5)),n20=int(t.sum()),tp=tp,tn=tn,fp=fp,fn=fn,
                balanced_accuracy=float((y[t].mean()+(~y[~t]).mean())/2),accuracy=float(np.mean(t==y)),baseline_accuracy=float(np.mean(~t)))

info['additional_labeled_records']=metrics(newrows)
info['additional_folders']=sorted(set(Path(r['path']).parent.name for r in newrows))
info['supplemental_snapshot']=metrics(rows+newrows)
(BASE/'work'/'additional_predictions.json').write_text(json.dumps(newrows,ensure_ascii=False),encoding='utf-8')

# Exercise the exact UI methods, without opening a window or starting any worker.
boundaries=[]
for e in [0,2,-2,2.4,-2.4,2.5,2.50001,5,-5,5.49,5.5,10,-10,10.5,10.50001]:
    it=ReviewItem(0,'synthetic',anno=0.,cake=100.,pred=e,status='完成')
    boundaries.append(dict(error=e,err_i=it.err_i,text=ReviewWindow._err_text(None,it),color=ReviewWindow._row_color(None,it).name()))
info['ui_boundaries']=boundaries
info['baseline_probe']={}
for anno,cake in [(10,20),(None,20),(None,None)]:
    it=ReviewItem(0,'synthetic',anno=anno,cake=cake,pred=12.,status='完成')
    info['baseline_probe'][str((anno,cake))]=dict(baseline=it.baseline,err=it.err)

# Inputs are synthetic in-memory data or scratch files under work/, never under F:.
tmp=BASE/'work'/'image_probes';tmp.mkdir(exist_ok=True)
improbes={}
for mode in ['L','RGBA','LA','P','RGB','CMYK']:
    im=Image.new(mode,(19,11))
    path=tmp/(mode+('.jpg' if mode=='CMYK' else '.png'))
    im.save(path)
    decoded=imread_bgr(path)
    improbes[mode]=dict(shape=decoded.shape,dtype=str(decoded.dtype),contiguous=decoded.flags.c_contiguous)
im=Image.new('RGB',(19,11));ex=Image.Exif();ex[274]=6
path=tmp/'rotated.jpg';im.save(path,exif=ex)
improbes['EXIF6']=dict(raw_size=[19,11],decoded_shape=imread_bgr(path).shape,
                           box_transform_present=False)
path=tmp/'bad.jpg';path.write_bytes(b'not a jpeg')
try:
    imread_bgr(path)
    improbes['bad_jpg']='unexpected success'
except Exception as ex:
    improbes['bad_jpg']=type(ex).__name__
for box in [(500.,500.,10.,10.),(5.5,5.5,0.,0.)]:
    try:
        views=backend.prepare_views(np.zeros((32,32,3),np.uint8),box)
        improbes['box_'+str(box)]=[x.shape for _,x in views]
    except Exception as ex:
        improbes['box_'+str(box)]=f'{type(ex).__name__}: {ex}'
info['image_and_box_probes']=improbes
(OUT/'audit_inference.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:info[k] for k in ['sample50','additional_labeled_records','supplemental_snapshot','image_and_box_probes']},ensure_ascii=False,indent=2),flush=True)
