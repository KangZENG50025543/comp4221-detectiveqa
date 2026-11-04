from pathlib import Path
import sys,json,platform,subprocess,importlib.metadata as metadata,os,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from detectiveqa import ROOT,write_json
import torch
report={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
 'packages':{p:metadata.version(p) for p in ['torch','transformers','numpy','requests','huggingface-hub','safetensors']},
 'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
 'cache_environment':{k:os.environ.get(k) for k in ['HF_HOME','HF_HUB_CACHE','PIP_CACHE_DIR','TORCH_HOME','TEMP','TMP']},
 'd_free_bytes':shutil.disk_usage(ROOT).free}
if torch.cuda.is_available():
    device=torch.cuda.get_device_properties(0)
    report.update(gpu_name=device.name,gpu_memory_bytes=device.total_memory,compute_capability=list(torch.cuda.get_device_capability(0)))
    x=torch.tensor([1.,2.,3.],device='cuda');report['gpu_arithmetic_check']=float((x*x).sum())==14.0
    torch.cuda.synchronize()
else:report['gpu_arithmetic_check']=False
write_json(ROOT/'reports/environment.json',report)
print(json.dumps(report,ensure_ascii=True,indent=2))
if not report['gpu_arithmetic_check']:raise SystemExit('GPU check failed')
