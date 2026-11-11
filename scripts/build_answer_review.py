"""Build local-only response/evidence review from locked outputs; never reads gold."""
import argparse
import html
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from detectiveqa import read_json, write_json
from prepare_answer_diagnostic import sha
from score_answer_diagnostic import checked_protocol

RUBRIC = {
    'scope': 'Judge ONLY support by supplied fragments, not agreement with a reference answer. No labels, condition names or correctness scores supplied. AI review is not human ground truth.',
    'answer_support': {'full': 'Visible evidence supports the stated answer and all necessary entity links',
        'partial': 'Some relevant support but an essential link or distinguishing fact is absent',
        'none': 'No relevant support, or visible evidence contradicts the answer',
        'unclear': 'Cannot resolve support or option uniqueness from supplied evidence',
        'not_applicable': 'The response abstains; do not treat the missing answer as a factual claim'},
    'explanation_support': 'Use the same full/partial/none/unclear scale for material factual claims in brief_reason. A justified statement that the evidence is insufficient can be fully supported even when the answer is unsupported. Use not_applicable only for empty/no factual explanation.',
    'citation_entailment': 'For the actual cited pid/quote pairs, do they jointly support the answer and its material justification? full/partial/none/unclear; no_citations if empty. A matching quote is not sufficient by itself.',
    'instructions': 'Read only this packet. Do not search or inspect reference answers/full original context/other reviews. State one short concrete note citing visible pids for each item. Distinguish known absent evidence from a claim that no evidence exists in the source. Flag potential question ambiguity or entity mismatch as tentative, not a corrected answer label.',
    'output_schema': {'item_id': 'copy', 'answer_support': 'full|partial|none|unclear|not_applicable',
        'explanation_support': 'full|partial|none|unclear|not_applicable',
        'citation_entailment': 'full|partial|none|unclear|no_citations',
        'possible_question_issue': 'yes|no|unclear', 'note': 'brief, concrete, no invented facts'},
}


def select(field, title, options):
    return f'<label for="{field}">{title}</label><select id="{field}" data-field><option value="">尚未填写</option>' + ''.join(f'<option value="{v}">{t}</option>' for v,t in options) + '</select>'


def build(run):
    checked_protocol(run)
    frozen = read_json(run/'responses.lock.json')
    for name, digest in frozen['outputs'].items():
        if sha(run/name) != digest:
            raise ValueError('Changed response')
    out = run/'answer_review'
    if out.exists():
        raise FileExistsError('Do not overwrite review work')
    out.mkdir()
    cases = {}
    for meta in read_json(run/'condition_key.json'):
        item = read_json(run/'items'/f'{meta["item_id"]}.json')
        raw_response = read_json(run/'responses'/f'{meta["item_id"]}.json')
        # Reviewers judge support themselves, without the answerer's self-rating.
        response = {k: raw_response[k] for k in ('answer','abstain','citations','brief_reason')}
        case = cases.setdefault(meta['qid'], {'qid': meta['qid'], 'dataset': meta['dataset'], 'group_id': meta['group_id'],
            'question': item['question'], 'options': item['options'], 'variants': []})
        case['variants'].append({'item_id': item['item_id'], 'evidence': item['evidence'], 'response': response})
    case_list = list(cases.values())
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'response_lock_sha256': sha(run/'responses.lock.json'),
        'reference_answers_loaded': False, 'condition_names_visible': False, 'correctness_scores_visible': False,
        'answerer_sufficiency_self_rating_visible': False,
        'cases': [{'qid': c['qid'], 'dataset': c['dataset'], 'group_id': c['group_id'], 'items': [v['item_id'] for v in c['variants']]} for c in case_list],
        'content_handling': 'Contains copyrighted source excerpts and actual AI answers; local only, never publish packet/HTML.',
        'human_ratings': 'All blank; actual independent review required',
        'blinding_limit': 'No-evidence condition is visually identifiable. A/C labels hidden, not all study features.'}
    write_json(out/'manifest.json', manifest)
    for dataset in ('detectiveqa','musique','quality'):
        write_json(out/f'ai_review_{dataset}.json', {'rubric': RUBRIC, 'cases': [c for c in case_list if c['dataset']==dataset]})
    write_json(out/'rubric.json', RUBRIC)
    fields = {}
    sections = []
    supports = [('full','充分支持'),('partial','只支持一部分／缺少连接'),('none','不支持或相矛盾'),('unclear','无法判断'),('not_applicable','不适用')]
    cites = supports[:-1]+[('no_citations','没有引用')]
    issues = [('yes','有疑点'),('no','未发现'),('unclear','无法判断')]
    for n,c in enumerate(case_list,1):
        options = ''.join(f'<li><b>{html.escape(k)}</b> {html.escape(v)}</li>' for k,v in (c['options'] or {}).items())
        variants = []
        for v in c['variants']:
            iid, r = v['item_id'], v['response']
            fields[iid] = {'answer_support':None,'explanation_support':None,'citation_entailment':None,'possible_question_issue':None,'note':''}
            text = '\n\n'.join('\n'.join(f'[{f["pid"]}] {f["text"]}' for f in ch['fragments']) for ch in v['evidence'])
            quotes = '\n'.join(f'[{q["pid"]}] {q["quote"]}' for q in r['citations']) or '没有引用'
            variants.append(f'<section class="variant"><h3>材料组 {iid[-1]}</h3><p><b>实际回答：</b>{html.escape(str(r["answer"]) if r["answer"] is not None else "未作答")}</p><p><b>简短依据：</b>{html.escape(r["brief_reason"])}</p>'
                +f'<details open><summary>模型引用的原句</summary><pre>{html.escape(quotes)}</pre></details>'
                +f'<details><summary>展开模型实际看到的全部材料</summary><pre>{html.escape(text or "本条件未提供任何材料。")}</pre></details>'
                +select(iid+'_answer_support','给定材料能否支撑这个答案？',supports)
                +select(iid+'_explanation_support','简短依据里的关键事实是否有支持？',supports)
                +select(iid+'_citation_entailment','它实际引用的内容，能否支撑答案与解释？',cites)
                +select(iid+'_possible_question_issue','是否发现题意、选项或实体连接的疑点？',issues)
                +f'<label for="{iid}_note">具体依据或缺口（写段号；不确定就说明）</label><textarea id="{iid}_note" data-field></textarea></section>')
        sections.append(f'<article id="case{n}"><h2>{n:02d} · {html.escape(c["dataset"])}</h2><p class="qid">{html.escape(c["qid"])}</p><p class="question">{html.escape(c["question"])}</p><ul>{options}</ul>'+''.join(variants)+'</article>')
    template={'schema_version':1,'packet_manifest_sha256':sha(out/'manifest.json'),'reviewer':'','independent_review_confirmed':False,'exported_utc':None,'ratings':fields}
    write_json(out/'empty_human_review.json',template)
    embedded=json.dumps(template,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>答案与证据复核</title><style>
body{max-width:1000px;margin:24px auto;padding:0 24px 70px;background:#f6f7f3;color:#19352c;font:16px/1.8 "Microsoft YaHei",sans-serif}h1{font-size:30px}article{padding:24px;margin:30px 0;border:1px solid #cddbd1;background:white;border-radius:10px}h2{border-bottom:2px solid #dae2da}.variant{border-top:1px dashed #b2c7b6;margin-top:22px;padding-top:8px}.question{font-size:19px;font-weight:bold}.qid{font-size:13px;color:#52695f}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f4ed;padding:12px;font:15px/1.9 "Microsoft YaHei",sans-serif}label{display:block;margin-top:16px;font-weight:bold}input,select,textarea,button{font:inherit;padding:8px;border:1px solid #8ca795;border-radius:5px;max-width:100%;box-sizing:border-box}select,textarea{width:100%}textarea{min-height:80px}.note{background:#fff1c9;padding:14px}.toolbar{position:sticky;top:0;background:#e7eee5;padding:12px;z-index:2}button{background:#20583c;color:white;cursor:pointer}summary{cursor:pointer;font-weight:bold}nav{display:flex;flex-wrap:wrap;gap:12px}nav a{color:#20583c}#status{font-size:14px}</style></head><body>
<h1>答对之外，还要看证据</h1><p>12 道开发题 · 36 份真实 AI 回复 · 人工评分全部留空</p>
<p>两人先各自复核并导出，第三人再处理分歧，轮换角色。这里只判断给定材料能否支撑回答，不展示标准答案或方法名称。没有材料的条件可以看出来，不能称完全盲评。</p>
<p>“充分支持”要求关键连接齐全；只看到相关词语、人物同名或半条推理链，选“部分支持”或“无法判断”。引用确实存在，不等于它证明了结论。模型拒答时，答案支持选“不适用”；说明材料缺什么仍可单独评价。未实际阅读的项目请留空。</p>
<div class="note">本页含原文，只供本地小组使用，勿公开上传。无网络请求，不会自动保存，离开前导出。这里的判断由你亲自填写，不能把AI审阅记录当成人评。</div>
<label for="reviewer">你的姓名或组内代号</label><input id="reviewer" data-field autocomplete="off"><label><input type="checkbox" id="independent" data-field> 我独立完成这些判断，尚未参考其他人的评分或标准答案。</label>
<div class="toolbar"><button id="export">导出我的评分</button> <label style="display:inline">恢复评分 <input type="file" id="restore" accept="application/json,.json"></label><div id="status">尚未填写</div></div><nav>'''+''.join(f'<a href="#case{n}">{n:02d}</a>' for n in range(1,13))+'</nav>'+''.join(sections)+'''<script id="template" type="application/json">'''+embedded+'''</script><script>
"use strict";
const template=JSON.parse(document.getElementById('template').textContent);
const allowed={answer_support:['full','partial','none','unclear','not_applicable'],explanation_support:['full','partial','none','unclear','not_applicable'],citation_entailment:['full','partial','none','unclear','no_citations'],possible_question_issue:['yes','no','unclear']};
function collect(){const r=JSON.parse(JSON.stringify(template));r.reviewer=document.getElementById('reviewer').value.trim();r.independent_review_confirmed=document.getElementById('independent').checked;r.exported_utc=new Date().toISOString();for(const [id,row] of Object.entries(r.ratings)){for(const k of Object.keys(row)){const val=document.getElementById(id+'_'+k).value;row[k]=k==='note'?val:val||null;}}return r;}
function progress(){const r=collect();let n=0;for(const row of Object.values(r.ratings))if(Object.keys(allowed).every(k=>row[k]!==null))n++;document.getElementById('status').textContent=`已填完四项判断：${n}/36份回复。离开前请导出。`;}
document.querySelectorAll('[data-field]').forEach(e=>e.addEventListener('input',progress));
document.getElementById('export').onclick=()=>{const blob=new Blob([JSON.stringify(collect(),null,2)],{type:'application/json;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='answer_review_'+new Date().toISOString().replace(/[:.]/g,'-')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
document.getElementById('restore').onchange=async e=>{try{const f=e.target.files[0];if(!f)return;const r=JSON.parse(await f.text());if(r.packet_manifest_sha256!==template.packet_manifest_sha256||typeof r.reviewer!=='string'||typeof r.independent_review_confirmed!=='boolean'||!r.ratings||Object.keys(r.ratings).sort().join('|')!==Object.keys(template.ratings).sort().join('|'))throw new Error('不是本轮复核包的评分');for(const [id,row] of Object.entries(r.ratings)){if(!row||Object.keys(row).sort().join('|')!==Object.keys(template.ratings[id]).sort().join('|')||typeof row.note!=='string')throw new Error('字段不匹配');for(const [k,opts] of Object.entries(allowed))if(row[k]!==null&&!opts.includes(row[k]))throw new Error('评分值不合法');}document.getElementById('reviewer').value=r.reviewer;document.getElementById('independent').checked=r.independent_review_confirmed;for(const [id,row] of Object.entries(r.ratings))for(const [k,v] of Object.entries(row))document.getElementById(id+'_'+k).value=v??'';progress();}catch(err){document.getElementById('status').textContent='恢复失败：'+err.message;}};
</script></body></html>'''
    (out/'answer_review.html').write_text(page,encoding='utf8')
    print(json.dumps({'html':str(out/'answer_review.html'),'cases':len(cases),'responses':len(fields),'human_ratings':0,'gold_loaded':False}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run-name',default='answer_diagnostic_dev_20261111_01');args=ap.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in {'.','..'}:
        raise ValueError('Invalid run name')
    build(ROOT/'runs'/args.run_name)
