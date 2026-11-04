"""Download official pinned files; local research use only, no API generation."""
from pathlib import Path
import argparse, hashlib, json, os, time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import requests

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/protocol.json').read_text(encoding='utf8'))

def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')

def fetch(repo,revision,name,dest,kind='model',expected_sha=None):
    prefix='datasets/' if kind=='dataset' else ''
    url=f'https://huggingface.co/{prefix}{repo}/resolve/{revision}/{quote(name,safe="/")}?download=true'
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        digest=hashlib.sha256(dest.read_bytes()).hexdigest()
        if expected_sha and digest != expected_sha:raise ValueError(f'Existing file hash mismatch: {dest}')
        return {'path':str(dest.relative_to(ROOT)),'source':url,'sha256':digest,'bytes':dest.stat().st_size}
    partial=dest.with_name(dest.name+'.partial')
    for attempt in range(3):
        try:
            with requests.get(url,stream=True,timeout=(30,120)) as response:
                response.raise_for_status(); digest=hashlib.sha256(); n=0; last=time.monotonic()
                with partial.open('wb') as f:
                    for chunk in response.iter_content(4*1024*1024):
                        if not chunk:continue
                        f.write(chunk);digest.update(chunk);n+=len(chunk)
                        if time.monotonic()-last>25:
                            print(f'{repo}/{name}: {n/1024**2:.0f} MiB downloaded',flush=True);last=time.monotonic()
            if expected_sha and digest.hexdigest()!=expected_sha:raise ValueError(f'Download SHA256 mismatch: {name}')
            partial.replace(dest)
            print(f'OK {repo}/{name}: {n} bytes',flush=True)
            return {'path':str(dest.relative_to(ROOT)),'source':url,'sha256':digest.hexdigest(),'bytes':n}
        except requests.RequestException:
            if attempt==2:raise
            time.sleep(2*(attempt+1))

def dataset():
    repo=CFG['dataset'];rev=CFG['dataset_revision']
    response=requests.get(f'https://huggingface.co/api/datasets/{repo}/revision/{rev}',timeout=30);response.raise_for_status();meta=response.json()
    save_json(ROOT/'data/raw/dataset_metadata.json',meta)
    names=[s['rfilename'] for s in meta['siblings']]
    anno=[n for n in names if n.startswith('anno_data_zh/human_anno/') and n.endswith('.json')]
    ids={Path(n).stem for n in anno}
    novels=[n for n in names if n.startswith('novel_data_zh/') and Path(n).name.split('-')[0] in ids]
    selected=['README.md']+anno+novels
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(lambda n:fetch(repo,rev,n,ROOT/'data/raw'/n,'dataset'),selected))
    save_json(ROOT/'reports/dataset_download_manifest.json',{'revision':rev,'files':records})

def models():
    manifests=[]
    for key in ['embedding','reranker']:
        spec=CFG[key];repo=spec['repo'];rev=spec['revision']
        response=requests.get(f'https://huggingface.co/api/models/{repo}/tree/{rev}',timeout=30);response.raise_for_status();tree=response.json()
        allow={'README.md','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model','pytorch_model.bin' if key=='embedding' else 'model.safetensors'}
        records=[]
        for entry in tree:
            name=entry['path']
            if name not in allow:continue
            records.append(fetch(repo,rev,name,ROOT/'models'/key/name,expected_sha=entry.get('lfs',{}).get('oid')))
        manifests.append({'role':key,'repo':repo,'revision':rev,'files':records})
        save_json(ROOT/'reports/model_download_manifest.json',manifests)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('asset',choices=['dataset','models']);a=p.parse_args()
    (dataset if a.asset=='dataset' else models)()
