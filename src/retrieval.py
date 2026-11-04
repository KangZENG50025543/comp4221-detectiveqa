"""BGE dense retrieval and pairwise reranking, with auditable character spans."""
import hashlib,json
from detectiveqa import query_text

def render_chunk(spans,by_id):
    return '\n'.join(f'[{s["pid"]}]'+by_id[s['pid']][s['start']:s['end']] for s in spans)

def make_chunks(paragraphs,tokenizer,max_tokens=512):
    by_id={p['pid']:p['text'] for p in paragraphs};chunks=[];current=[]
    def tokens(spans):return len(tokenizer.encode(render_chunk(spans,by_id),add_special_tokens=True))
    def flush():
        nonlocal current
        if current:
            text=render_chunk(current,by_id)
            chunks.append({'id':len(chunks),'text':text,'spans':current,'bge_tokens':tokens(current)})
            current=[]
    for p in paragraphs:
        text=p['text'];span={'pid':p['pid'],'start':0,'end':len(text)}
        if tokens([span])<=max_tokens:
            if current and tokens(current+[span])>max_tokens:flush()
            current.append(span)
        else:
            flush();start=0
            while start<len(text):
                tail=text[start:]
                offsets=tokenizer(tail,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
                if not offsets:raise ValueError('Tokenizer produced no offsets')
                count=min(len(offsets),max_tokens-16)
                end=len(text) if count==len(offsets) else start+offsets[count-1][1]
                fragment={'pid':p['pid'],'start':start,'end':end}
                while tokens([fragment])>max_tokens and end>start:
                    end-=1;fragment['end']=end
                if end<=start:raise ValueError('Cannot fit paragraph fragment')
                current=[fragment];flush();start=end
    flush()
    if any(c['bge_tokens']>max_tokens for c in chunks):raise AssertionError('Chunk token cap exceeded')
    # Every source character appears once. This catches overlap, omitted whitespace and offset errors.
    for p in paragraphs:
        spans=[s for c in chunks for s in c['spans'] if s['pid']==p['pid']]
        cursor=0
        for s in spans:
            if s['start']!=cursor:raise AssertionError('Chunk coverage gap or overlap')
            cursor=s['end']
        if cursor!=len(p['text']):raise AssertionError('Incomplete paragraph coverage')
    return chunks

def chunk_fingerprint(chunks,revision):
    return hashlib.sha256(json.dumps({'revision':revision,'texts':[c['text'] for c in chunks]},ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def embed_texts(texts,model,tokenizer,batch_size=8,max_length=8192):
    import torch
    import numpy as np
    vectors=[]
    for start in range(0,len(texts),batch_size):
        batch=texts[start:start+batch_size]
        lengths=[len(tokenizer.encode(t,add_special_tokens=True)) for t in batch]
        if max(lengths)>max_length:raise ValueError('Embedding input exceeds model limit; no silent truncation')
        inputs=tokenizer(batch,padding=True,truncation=False,return_tensors='pt').to(model.device)
        with torch.inference_mode():
            output=model(**inputs).last_hidden_state[:,0]
            vectors.append(torch.nn.functional.normalize(output.float(),p=2,dim=1).cpu().numpy())
    return np.concatenate(vectors,axis=0)

def rerank_scores(query,texts,model,tokenizer,batch_size=4,max_length=8192):
    import torch
    scores=[]
    for start in range(0,len(texts),batch_size):
        batch=texts[start:start+batch_size]
        lengths=[len(tokenizer.encode(query,t,add_special_tokens=True)) for t in batch]
        if max(lengths)>max_length:raise ValueError('Reranker input exceeds limit; no silent truncation')
        inputs=tokenizer([query]*len(batch),batch,padding=True,truncation=False,return_tensors='pt').to(model.device)
        with torch.inference_mode():
            logits=model(**inputs).logits.float().view(-1)
            scores.extend(logits.cpu().tolist())
    return scores
