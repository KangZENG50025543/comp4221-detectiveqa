"""A local reading packet with blank human audit fields; no generated ratings."""
from pathlib import Path
import sys,html,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from detectiveqa import ROOT,read_json,read_jsonl,write_json

inputs=read_jsonl(ROOT/'data/processed/dev_inputs.jsonl')[:10]
gold={q['qid']:q for q in read_jsonl(ROOT/'data/restricted/dev_gold.jsonl')}
e=html.escape;cards=[]
for q in inputs:
    reference=gold[q['qid']];paras={p['pid']:p['text'] for p in read_json(ROOT/f'data/processed/books/{q["book_id"]}.json')}
    options=''.join(f'<li><b>{key}</b> {e(q["options"][key])}</li>' for key in 'ABCD')
    clues=''.join(f'<details><summary>原文段落 [{pid}]</summary><p>{e(paras[pid])}</p></details>' for pid in reference['gold_pids'])
    boundary=''.join(f'<h4>[{pid}] {"揭示边界（普通检索不可见）" if pid==q["cutoff_exclusive"] else "安全前缀最后一段"}</h4><p>{e(paras[pid])}</p>' for pid in [q['cutoff_exclusive']-1,q['cutoff_exclusive']])
    cards.append(f'''<article data-qid="{q['qid']}"><h2>{q['qid']} · 小说 {q['book_id']}</h2>
    <p>{e(q['question'])}</p><ul>{options}</ul><p>数据集标注答案：<b>{reference['answer']}</b>。这不是我们人工确认的答案。</p>
    <h3>数据集标注的显式线索</h3>{clues}
    <details><summary>查看前缀截止位置及揭示段落</summary>{boundary}</details>
    <label>核验结果 <select><option value="">未填写</option><option>位置和线索支持均已核实</option><option>位置有疑问</option><option>线索支持有疑问</option><option>无法判断</option></select></label>
    <label>核验者 <input class="reviewer" placeholder="由实际核验者填写" /></label>
    <label>依据或疑问 <textarea rows="3"></textarea></label></article>''')
template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DetectiveQA 开发集人工核验</title><style>body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:1000px;margin:32px auto;padding:0 24px;color:#16202a;background:#f5f6f8}h1,h2,h3,h4{color:#111}article{background:white;border:1px solid #d9d9d9;border-radius:8px;padding:24px;margin:24px 0}summary,button{cursor:pointer}details{border-bottom:1px solid #e3e3e3;padding:9px 0}label{display:block;margin-top:16px}input,select,textarea{box-sizing:border-box;max-width:100%;padding:8px;font:inherit}textarea{width:100%}button{padding:12px 18px;background:#203e56;color:white;border:0;border-radius:5px}p{white-space:pre-wrap}.note{color:#54626e}</style>
<h1>DetectiveQA 开发集人工核验</h1><p>这里只使用开发集的 10 道真实题。请核对显式线索与原文是否对应、前缀截止是否合理。所有人工字段初始留空；本页面没有代替人类判断。</p>
<p class="note">数据来自固定版本 Phospheneser/DetectiveQA，仅供本组本地研究；含原小说片段，请勿公开上传。正文结构已自动检查，语义尚待人工核验。保存按钮会导出你实际填写的 JSON；关闭前请先导出。</p>
<button onclick="save()">导出已填写的核验记录</button>__CARDS__
<script>function save(){const records=[...document.querySelectorAll('article')].map(a=>({qid:a.dataset.qid,status:a.querySelector('select').value||null,reviewer:a.querySelector('input').value||null,note:a.querySelector('textarea').value||null}));const blob=new Blob([JSON.stringify({exported_at:new Date().toISOString(),records},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='detectiveqa_manual_audit.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}</script></html>'''
(ROOT/'reports/dev_manual_audit.html').write_text(template.replace('__CARDS__','\n'.join(cards)),encoding='utf8')
write_json(ROOT/'reports/manual_audit_status.json',{'sample_ids':[q['qid'] for q in inputs],'human_completed':False,'ratings':[]})
print('Created local 10-question reading packet; all human fields are blank.')
