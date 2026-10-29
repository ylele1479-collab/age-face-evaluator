"""Portable entry point; model and user data paths are relative to this package."""
from pathlib import Path
import os,sys,runpy
ROOT=Path(__file__).resolve().parent
os.environ['LOCALAPPDATA']=str(ROOT/'data')
os.environ['DEEPFACE_HOME']=str(ROOT/'models/deepface-home')
os.environ['HF_HOME']=str(ROOT/'models/huggingface')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
sys.path.insert(0,str(ROOT/'source'))
mode=sys.argv[1] if len(sys.argv)>1 else 'main'
target=ROOT/'source'/('review_main.py' if mode=='review' else 'main.py')
sys.argv=[str(target),*sys.argv[2:]]
if mode=='audit':
    target=ROOT/'audit/audit_accuracy.py'
elif mode=='selfcheck':
    target=ROOT/'selfcheck.py'
runpy.run_path(str(target),run_name='__main__')
