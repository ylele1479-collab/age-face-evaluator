"""Check the bundled runtime, GUI dependencies and local model inference."""
from pathlib import Path
import os,json,sys,time
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('OMP_NUM_THREADS','2')
started=time.monotonic()
import numpy as np
import cv2, torch, tensorflow, transformers, openpyxl
from PySide6.QtWidgets import QApplication
from age_face_tool.config import models_dir
from age_face_tool.core.age_backend import DeepFaceAgeBackend
from age_face_tool.core.vit_backend import ViTAgeBackend
from age_face_tool.core.gender_backend import GenderBackend

assert str(ROOT/'runtime').lower() in str(sys.prefix).lower(),sys.prefix
assert os.environ['HF_HUB_OFFLINE']=='1'
assert (models_dir()/'face_detection_yunet_2023mar.onnx').is_file()
detector=cv2.FaceDetectorYN.create(str(models_dir()/'face_detection_yunet_2023mar.onnx'),'',(320,320))
detector.detect(np.zeros((320,320,3),np.uint8))
sample=np.full((160,160,3),128,np.uint8)
results={}
for name,cls in [('deepface',DeepFaceAgeBackend),('vit',ViTAgeBackend)]:
    model=cls()
    ages,dist=model.predict_batch([sample])
    assert len(ages)==1 and np.isfinite(ages).all() and 0<=ages[0]<=100,(name,ages)
    assert np.isfinite(dist).all(),name
    results[name]={'output_shape':list(dist.shape),'dummy_prediction':float(ages[0])}
    print('PASS',name,flush=True)
gender=GenderBackend().predict_patches([sample])
assert len(gender)==2 and np.isfinite(gender).all()
print('PASS gender',flush=True)
app=QApplication.instance() or QApplication([])
from age_face_tool.ui.main_window import MainWindow
window=MainWindow()
window.show();app.processEvents();window.close();app.processEvents()
results.update(runtime=sys.version,qt='MainWindow created offscreen',
    seconds=round(time.monotonic()-started,2),
    note='Synthetic input validates loading and numerical output, not accuracy or live desktop synchronization.')
print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)
