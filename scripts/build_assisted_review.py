"""Create an AI-prefilled verification page; human confirmations always start empty."""
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

FIELDS = ['answer_support', 'explanation_support', 'citation_entailment', 'possible_question_issue']
LABELS = {'answer_support':'材料能否支撑答案', 'explanation_support':'解释中的关键事实是否有支持',
          'citation_entailment':'实际引用能否支撑结论', 'possible_question_issue':'题目本身是否有疑点'}
SUPPORT = {'full':'充分支持', 'partial':'部分支持／仍缺连接', 'none':'不支持或相矛盾', 'unclear':'无法判断', 'not_applicable':'不适用'}
OPTIONS = {'answer_support':SUPPORT, 'explanation_support':SUPPORT,
           'citation_entailment':{**{k:v for k,v in SUPPORT.items() if k!='not_applicable'}, 'no_citations':'没有引用'},
           'possible_question_issue':{'yes':'有疑点', 'no':'未发现', 'unclear':'无法判断'}}


def load_draft(run, draft_path):
    checked_protocol(run)
    lock = read_json(run/'responses.lock.json')
    for name, digest in lock['outputs'].items():
        if digest is None or sha(run/name) != digest:
            raise ValueError('This review page requires all immutable replies')
    review = run/'answer_review'
    manifest = read_json(review/'manifest.json')
    if manifest['response_lock_sha256'] != sha(run/'responses.lock.json'):
        raise ValueError('Original review refers to different replies')
    cases, previous = [], {}
    for dataset in ('detectiveqa','musique','quality'):
        cases += read_json(review/f'ai_review_{dataset}.json')['cases']
        for p in read_json(review/f'ai_support_{dataset}.json')['items']:
            if p['item_id'] in previous:
                raise ValueError('Duplicated previous review')
            previous[p['item_id']] = p
    variants = {v['item_id']:v for c in cases for v in c['variants']}
    expected = {iid for c in manifest['cases'] for iid in c['items']}
    if len(variants)!=36 or set(variants)!=expected or set(previous)!=expected:
        raise ValueError('Review item set changed')
    for iid,v in variants.items():
        item=read_json(run/'items'/f'{iid}.json')
        raw=read_json(run/'responses'/f'{iid}.json')
        if v['evidence']!=item['evidence'] or v['response']!={k:raw[k] for k in ('answer','abstain','citations','brief_reason')}:
            raise ValueError('Review material differs from immutable input/response')
    draft=read_json(draft_path)
    if (draft.get('reviewer_type')!='AI' or draft.get('human_confirmations')!=0
            or draft.get('reference_answers_seen') is not False or draft.get('condition_names_seen') is not False):
        raise ValueError('Expected AI draft without human confirmations or gold access')
    rows=draft['items']
    if len(rows)!=len(expected) or {r['item_id'] for r in rows}!=expected:
        raise ValueError('Incomplete or duplicated AI draft')
    for r in rows:
        iid=r['item_id']
        if any(r.get(f) not in OPTIONS[f] for f in FIELDS):
            raise ValueError('Unknown rating')
        if r.get('review_priority') not in {'high','normal'} or any(not isinstance(r.get(f),str) or not r[f].strip() for f in ('note','check_instruction')):
            raise ValueError('Missing explanation or review instruction')
        changed=[f for f in FIELDS if r[f]!=previous[iid][f]]
        if sorted(r['changed_from_previous'])!=sorted(changed):
            raise ValueError('Claimed changes do not match previous AI judgments')
        if (r['answer_support']=='not_applicable')!=variants[iid]['response']['abstain']:
            raise ValueError('Abstention applicability mismatch')
        if (r['citation_entailment']=='no_citations')!=(not variants[iid]['response']['citations']):
            raise ValueError('Citation applicability mismatch')
    return cases,draft,previous


def build(run, draft_path, review_name):
    cases,draft,previous=load_draft(run,draft_path)
    out=run/'answer_review'/review_name
    if out.exists():raise FileExistsError('Do not overwrite any human verification work')
    out.mkdir()
    proposals={r['item_id']:r for r in draft['items']}
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(), 'review_mode':'ai_assisted_human_verification',
        'ai_draft_sha256':sha(draft_path),'response_lock_sha256':sha(run/'responses.lock.json'),
        'prior_review_manifest_sha256':sha(run/'answer_review/manifest.json'),
        'items':list(proposals),'question_count':len(cases),'item_count':len(proposals),
        'initial_human_confirmations':0,'independent_blind_review':False,
        'content_handling':'Local only: contains source excerpts. Export contains suggestions and actual human decisions, no original context.',
        'old_artifacts_preserved':True}
    write_json(out/'manifest.json',manifest)
    template={'schema_version':2,'review_mode':'ai_assisted_human_verification','independent_blind_review':False,
        'packet_manifest_sha256':sha(out/'manifest.json'),'ai_draft_sha256':sha(draft_path),
        'reviewer':'','reviewer_role':'current operator/exporter; each confirmation retains its own confirmed_by', 'exported_utc':None,'status':'not_started',
        'items':{iid:{'decision':None,'confirmed_at':None,'confirmed_by':None,'human_ratings':None,
            'draft_ratings':{f:p[f] for f in FIELDS},'human_note':'','history':[]} for iid,p in proposals.items()}}
    write_json(out/'empty_verification_state.json',template)
    snapshot={'reviewer_type':'AI','requested_model':draft['requested_model'],'items':draft['items']}
    write_json(out/'ai_suggestions.json',snapshot)
    articles=[]
    for n,c in enumerate(cases,1):
        opts=''.join(f'<li><b>{html.escape(k)}</b> {html.escape(v)}</li>' for k,v in (c['options'] or {}).items())
        cards=[]
        for v in c['variants']:
            iid=v['item_id'];p=proposals[iid];r=v['response']
            source='\n\n'.join('\n'.join(f'[{f["pid"]}] {f["text"]}' for f in ch['fragments']) for ch in v['evidence'])
            quotes='\n'.join(f'[{q["pid"]}] {q["quote"]}' for q in r['citations']) or '这份回答没有引用。'
            selectors=''
            for field in FIELDS:
                choices=''.join(f'<option value="{k}"'+(' selected' if k==p[field] else '')+f'>{label}</option>' for k,label in OPTIONS[field].items())
                selectors+=f'<label for="{iid}_{field}">{LABELS[field]}<small>AI建议：{OPTIONS[field][p[field]]}</small></label><select id="{iid}_{field}" data-item="{iid}">{choices}</select>'
            changed='、'.join(LABELS[f] for f in p['changed_from_previous'])
            change_note=f'<p class="change">本轮AI复查调整了：{changed}。旧记录保留。</p>' if changed else ''
            cards.append(f'<section class="card" id="{iid}" data-priority="{p["review_priority"]}"><h3>材料组 {iid[-1]}'+('<span class="badge">建议重点看</span>' if p['review_priority']=='high' else '')+'</h3>'
                +f'<p><b>实际回答：</b>{html.escape(str(r["answer"]) if r["answer"] is not None else "未作答")}</p><p><b>回答的依据：</b>{html.escape(r["brief_reason"])}</p>'
                +f'<details><summary>核对模型引用的原句</summary><pre>{html.escape(quotes)}</pre></details>'
                +f'<details><summary>展开这组实际提供的全部材料</summary><pre>{html.escape(source or "本组没有提供材料。")}</pre></details>'
                +f'<div class="ai"><b>AI 初评依据</b><p>{html.escape(p["note"])}</p><b>你重点核对这一点</b><p>{html.escape(p["check_instruction"])}</p>{change_note}</div>'
                +'<div class="ratings">'+selectors+'</div>'
                +f'<label for="{iid}_note">你的补充或修改理由（可留空）</label><textarea id="{iid}_note" data-item="{iid}" placeholder="这里只记录你实际填写的意见"></textarea>'
                +f'<div class="actions"><button type="button" data-confirm="{iid}">确认本组判断</button><button type="button" class="secondary" data-uncertain="{iid}">暂时拿不准</button><button type="button" class="plain" data-retract="{iid}">撤回确认</button></div><p class="item-status" id="{iid}_status">待你复核 · 当前显示的是AI建议</p></section>')
        articles.append(f'<article id="question{n}"><h2>{n:02d} · {html.escape(c["dataset"])}</h2><p class="qid">{html.escape(c["qid"])}</p><p class="question">{html.escape(c["question"])}</p><ul>{opts}</ul>'+''.join(cards)+'</article>')
    def embed(value):return json.dumps(value,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI先评，你来复核</title><style>
body{max-width:1040px;margin:28px auto;padding:0 22px 90px;background:#f5f6f2;color:#203c33;font:16px/1.8 "Microsoft YaHei",sans-serif}h1{font-size:31px;margin-bottom:8px}h2{border-bottom:2px solid #d9e2d8}h3{font-size:20px}.lead{font-size:18px}.notice{background:#fff0c8;padding:13px 17px;border-radius:8px}article{background:white;border:1px solid #d3dfd3;border-radius:12px;padding:24px;margin-top:30px}.card{border-top:1px dashed #bacbbb;padding:22px 0;scroll-margin-top:125px}.question{font-size:19px;font-weight:bold}.qid,small{font-size:13px;color:#5d7166}.ai{background:#eef4ea;border-left:4px solid #64936a;padding:13px 17px;margin:18px 0}.ai p{margin:6px 0 12px}.change{font-size:14px;color:#665126}.badge{display:inline-block;font-size:13px;background:#fff0c8;padding:2px 9px;margin-left:14px;border-radius:4px}.ratings{display:grid;grid-template-columns:1fr 1fr;gap:10px 18px;align-items:center}label{display:block;font-weight:bold}small{display:block;font-weight:normal}select,input,textarea,button{font:inherit;border:1px solid #91a795;border-radius:6px;padding:8px;box-sizing:border-box;max-width:100%}select{width:100%;background:white}textarea{width:100%;min-height:65px;margin:8px 0}button{background:#276145;color:white;cursor:pointer}.secondary{background:#f0e8d2;color:#5c4d25}.plain{background:white;color:#496251}.actions{display:flex;gap:10px;flex-wrap:wrap}.item-status{color:#6d6549;font-size:14px}.toolbar{position:sticky;top:0;background:#e5eee2;border:1px solid #ccd8c8;border-radius:6px;padding:12px;z-index:2;margin:16px 0}.toolbar label{display:inline;font-weight:normal}.toolbar p{margin:5px 0;font-size:14px}nav{display:flex;gap:16px;flex-wrap:wrap}nav a{color:#275b3d}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3ed;padding:13px;font:15px/1.9 "Microsoft YaHei",sans-serif}summary{cursor:pointer;font-weight:bold}details{margin:12px 0}.card.confirmed{border-left:3px solid #4a8a5d;padding-left:14px}.card.uncertain{border-left:3px solid #b18b3b;padding-left:14px}@media(max-width:650px){.ratings{grid-template-columns:1fr;gap:4px}article{padding:14px}.toolbar{position:static}}
</style></head><body><h1>AI先评，你来复核</h1><p class="lead">12道题 · 36份回答 · AI建议已填好，人工确认尚未开始</p>
<p>先看AI的判断和依据，再展开相关段落核对。认可就点“确认本组判断”；不认可直接修改选项后确认；拿不准可标成待讨论。无需从空表重新填。</p>
<p>下拉框当前显示AI建议。只有你点击确认，才记录为你的复核；确认后再修改，会自动恢复为待确认。这是“AI初评＋人工复核”，会如实记录这一过程。</p>
<div class="notice">页面含原文，仅在本地使用。不会自动保存，离开前请导出；下次可恢复记录。AI的理由也可能有误，重点核对它指出的段号和缺口。</div>
<label for="reviewer">当前复核人的姓名或组内代号（每次确认都会记录）</label><input id="reviewer" autocomplete="off" placeholder="实际复核人自行填写">
<div class="toolbar"><button type="button" id="export">导出复核记录</button> <label>恢复记录 <input id="restore" type="file" accept="application/json,.json"></label><p id="progress">已确认 0/36</p><p id="message" role="status">人工确认尚未开始。可以随时导出进度。</p><label><input id="only-priority" type="checkbox"> 只显示重点项目</label> <label><input id="only-open" type="checkbox"> 隐藏已确认项目</label></div><nav>'''+''.join(f'<a href="#question{n}">{n:02d}</a>' for n in range(1,len(cases)+1))+'</nav>'+''.join(articles)+'''
<script id="template" type="application/json">'''+embed(template)+'''</script><script id="suggestions" type="application/json">'''+embed(snapshot)+'''</script><script id="options" type="application/json">'''+embed(OPTIONS)+'''</script><script>
"use strict";
const template=JSON.parse(document.getElementById('template').textContent);
const suggestions=JSON.parse(document.getElementById('suggestions').textContent);
const ai=Object.fromEntries(suggestions.items.map(r=>[r.item_id,r]));
const options=JSON.parse(document.getElementById('options').textContent), fields=Object.keys(options);
let state=JSON.parse(JSON.stringify(template));
const clone=x=>JSON.parse(JSON.stringify(x));
const equal=(a,b)=>fields.every(k=>a?.[k]===b?.[k]);
function draftValues(id){return Object.fromEntries(fields.map(k=>[k,document.getElementById(id+'_'+k).value]));}
function sync(id){state.items[id].draft_ratings=draftValues(id);state.items[id].human_note=document.getElementById(id+'_note').value;}
function operator(){return document.getElementById('reviewer').value.trim();}
function requireOperator(){if(operator())return true;document.getElementById('message').textContent='请先填写实际复核人的姓名或代号，再记录判断。';return false;}
function history(id,action){state.items[id].history.push({action,at:new Date().toISOString(),actor:operator()||null,ratings:clone(state.items[id].draft_ratings),note:state.items[id].human_note});}
function changed(id){sync(id);const r=state.items[id];if(r.decision!==null){history(id,'edited_after_decision');r.decision=null;r.confirmed_at=null;r.confirmed_by=null;r.human_ratings=null;}renderStatus();}
function confirm(id){if(!requireOperator())return;sync(id);const r=state.items[id];r.decision=equal(r.draft_ratings,ai[id])?'accepted':'corrected';r.confirmed_at=new Date().toISOString();r.confirmed_by=operator();r.human_ratings=clone(r.draft_ratings);history(id,r.decision);renderStatus();}
function uncertain(id){if(!requireOperator())return;sync(id);const r=state.items[id];r.decision='uncertain';r.confirmed_at=null;r.confirmed_by=null;r.human_ratings=null;history(id,'uncertain');renderStatus();}
function retract(id){if(!requireOperator())return;sync(id);const r=state.items[id];history(id,'retracted');r.decision=null;r.confirmed_at=null;r.confirmed_by=null;r.human_ratings=null;renderStatus();}
function counts(){const rows=Object.values(state.items);return {confirmed:rows.filter(r=>['accepted','corrected'].includes(r.decision)).length,uncertain:rows.filter(r=>r.decision==='uncertain').length,pending:rows.filter(r=>r.decision===null).length};}
function renderStatus(){const c=counts();document.getElementById('progress').textContent=`已确认 ${c.confirmed}/36 · 待讨论 ${c.uncertain} · 尚未确认 ${c.pending}`;for(const [id,r] of Object.entries(state.items)){const el=document.getElementById(id);el.classList.toggle('confirmed',['accepted','corrected'].includes(r.decision));el.classList.toggle('uncertain',r.decision==='uncertain');document.getElementById(id+'_status').textContent=r.decision==='accepted'?`已由 ${r.confirmed_by} 确认：同意AI初评`:r.decision==='corrected'?`已由 ${r.confirmed_by} 确认：采用修改`:r.decision==='uncertain'?'待讨论：尚无最终人工判断':'待你复核 · 当前选项尚未记作人工判断';}filter();}
function filter(){for(const [id,r] of Object.entries(state.items)){const el=document.getElementById(id);el.hidden=(document.getElementById('only-priority').checked&&ai[id].review_priority!=='high')||(document.getElementById('only-open').checked&&['accepted','corrected'].includes(r.decision));}}
function collect(){const r=clone(state);r.reviewer=document.getElementById('reviewer').value.trim();r.exported_utc=new Date().toISOString();const c=counts();r.status=c.confirmed===36&&r.reviewer?'verified':c.uncertain?'needs_discussion':c.confirmed||Object.values(r.items).some(x=>x.human_note||x.history.length)?'in_progress':'not_started';r.counts=c;r.ai_suggestions=clone(suggestions);return r;}
document.querySelectorAll('[data-item]').forEach(el=>el.addEventListener('input',()=>changed(el.dataset.item)));
document.querySelectorAll('[data-confirm]').forEach(el=>el.onclick=()=>confirm(el.dataset.confirm));
document.querySelectorAll('[data-uncertain]').forEach(el=>el.onclick=()=>uncertain(el.dataset.uncertain));
document.querySelectorAll('[data-retract]').forEach(el=>el.onclick=()=>retract(el.dataset.retract));
document.getElementById('only-priority').onchange=filter;document.getElementById('only-open').onchange=filter;
document.getElementById('export').onclick=()=>{const r=collect();const blob=new Blob([JSON.stringify(r,null,2)],{type:'application/json;charset=utf-8'});const u=URL.createObjectURL(blob);const a=document.createElement('a');a.href=u;a.download='assisted_review_'+new Date().toISOString().replace(/[:.]/g,'-')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);document.getElementById('message').textContent=r.reviewer?'已导出当前进度。':'已导出草稿；请填写实际复核人后再导出完成记录。';};
const keysEqual=(v,keys)=>v&&typeof v==='object'&&!Array.isArray(v)&&Object.keys(v).sort().join('|')===keys.slice().sort().join('|');
const validRatings=v=>keysEqual(v,fields)&&fields.every(k=>Object.hasOwn(options[k],v[k]));
const validTime=t=>typeof t==='string'&&Number.isFinite(Date.parse(t));
function validateImport(r){if(!keysEqual(r,[...Object.keys(template),'counts','ai_suggestions'])||r.schema_version!==template.schema_version||r.packet_manifest_sha256!==template.packet_manifest_sha256||r.ai_draft_sha256!==template.ai_draft_sha256||r.review_mode!==template.review_mode||r.reviewer_role!==template.reviewer_role||r.independent_blind_review!==false||typeof r.reviewer!=='string'||!keysEqual(r.items,Object.keys(template.items)))throw new Error('不是这一轮AI辅助复核记录');if(JSON.stringify(r.ai_suggestions)!==JSON.stringify(suggestions))throw new Error('AI建议被更改，无法恢复');for(const [id,row] of Object.entries(r.items)){if(!keysEqual(row,Object.keys(template.items[id]))||![null,'accepted','corrected','uncertain'].includes(row.decision)||typeof row.human_note!=='string'||!Array.isArray(row.history))throw new Error('复核字段不合法');if(!validRatings(row.draft_ratings))throw new Error('评分值不合法');for(const h of row.history){if(!keysEqual(h,['action','at','actor','ratings','note'])||!['accepted','corrected','uncertain','retracted','edited_after_decision'].includes(h.action)||!validTime(h.at)||!validRatings(h.ratings)||typeof h.note!=='string'||!(typeof h.actor==='string'&&h.actor.trim()||h.actor===null&&h.action==='edited_after_decision'))throw new Error('操作历史字段不合法');}const done=['accepted','corrected'].includes(row.decision);if(done){if(!validRatings(row.human_ratings)||!equal(row.human_ratings,row.draft_ratings)||!validTime(row.confirmed_at)||typeof row.confirmed_by!=='string'||!row.confirmed_by.trim())throw new Error('确认状态与评分不一致');if((row.decision==='accepted')!==equal(row.human_ratings,ai[id]))throw new Error('同意或修改状态不匹配');}else if(row.human_ratings!==null||row.confirmed_at!==null||row.confirmed_by!==null)throw new Error('未确认项不能带人工定稿');}}
document.getElementById('restore').onchange=async e=>{try{const f=e.target.files[0];if(!f)return;const r=JSON.parse(await f.text());validateImport(r);state=clone(template);state.items=clone(r.items);document.getElementById('reviewer').value=r.reviewer;for(const [id,row] of Object.entries(state.items)){for(const k of fields)document.getElementById(id+'_'+k).value=row.draft_ratings[k];document.getElementById(id+'_note').value=row.human_note;}renderStatus();document.getElementById('message').textContent='记录已恢复。';}catch(err){document.getElementById('message').textContent='恢复失败：'+err.message;}};
renderStatus();
</script></body></html>'''
    (out/'verify_ai_review.html').write_text(page,encoding='utf8')
    summary={'created_utc':manifest['created_utc'],'review_mode':manifest['review_mode'],'questions':len(cases),'responses':len(proposals),
        'ai_initial_assessment_completed':True,'ai_draft_sha256':sha(draft_path),
        'changed_items_vs_prior_ai':sum(bool(p['changed_from_previous']) for p in proposals.values()),
        'changed_rating_fields_vs_prior_ai':sum(len(p['changed_from_previous']) for p in proposals.values()),
        'high_priority_items':sum(p['review_priority']=='high' for p in proposals.values()),
        'human_confirmations':0,'human_ratings':None,'independent_blind_review':False,
        'items':[{'item_id':p['item_id'],'review_priority':p['review_priority'],'changed_from_previous':p['changed_from_previous']} for p in proposals.values()],
        'note':'AI suggestions are complete, human verification is pending. Prior AI reviews, blank independent-review page, original responses and scores are preserved.'}
    write_json(out/'summary.json',summary)
    print(json.dumps({'page':str(out/'verify_ai_review.html'),**{k:summary[k] for k in ('responses','changed_items_vs_prior_ai','changed_rating_fields_vs_prior_ai','high_priority_items','human_confirmations')}},ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run-name',default='answer_diagnostic_dev_20261111_01');ap.add_argument('--review-name',default='assisted_20261111_02');args=ap.parse_args()
    for v in (args.run_name,args.review_name):
        if Path(v).name!=v or v in {'.','..'}:raise ValueError('Names must be single directory components')
    run=ROOT/'runs'/args.run_name
    build(run,run/'answer_review/ai_assisted_draft_20261111.json',args.review_name)
