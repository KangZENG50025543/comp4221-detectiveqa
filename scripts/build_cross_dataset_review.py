"""Build a local, anonymous evidence-review packet. Never reads labels or metrics."""
import argparse
import hashlib
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from detectiveqa import prefix_for_question

SEED = 'human-review-20261111:'


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(qid):
    return hashlib.sha256((SEED + qid).encode()).hexdigest()


def checkbox_select(name, label):
    return (f'<label for="{name}">{html.escape(label)}</label><select id="{name}" data-review-field>'
            '<option value="">尚未填写</option><option value="yes">是</option>'
            '<option value="no">否</option><option value="unclear">无法判断 / 需要更多核查</option></select>')


def make_packet(detective_run, cross_run):
    # Check completed output sentinels, but never load metrics or reference answers.
    for complete in (detective_run / 'summary.json', cross_run / 'retrieval/summary.json'):
        if not complete.exists():
            raise FileNotFoundError(f'Retrieval run not complete: {complete}')
    record_paths = [detective_run / 'retrieval_records.jsonl', cross_run / 'retrieval/retrieval_records.jsonl']
    outputs = jsonl(record_paths[0]) + jsonl(record_paths[1])
    dev = {q['qid']: q for q in jsonl(ROOT / 'data/processed/dev_inputs.jsonl')}
    extra = {q['qid']: q for q in jsonl(cross_run / 'prepared/inputs.jsonl')}
    split = load(ROOT / 'configs/split.json')
    by_qid = {}
    for r in outputs:
        qid = r['qid']
        dataset = r.get('dataset', 'detectiveqa')
        if dataset == 'detectiveqa':
            if qid not in dev or qid not in split['dev_ids']:
                raise ValueError('Non-development DetectiveQA question')
        elif qid not in extra or extra[qid].get('source_split') != 'dev':
            raise ValueError('Non-development supplemental question')
        item = by_qid.setdefault(qid, {'dataset': dataset, 'records': {}})
        if item['dataset'] != dataset or r['condition'] in item['records']:
            raise ValueError('Duplicate or inconsistent retrieval record')
        item['records'][r['condition']] = r
    selected = []
    for dataset in ('detectiveqa', 'musique', 'quality'):
        ids = sorted((qid for qid, q in by_qid.items() if q['dataset'] == dataset), key=rank)[:4]
        if len(ids) != 4:
            raise ValueError('Need four completed cases per dataset')
        selected.extend(ids)

    out = cross_run / 'human_review'
    if out.exists():
        raise FileExistsError('Review packet is immutable; do not overwrite reviewer work')
    cases, keys = [], []
    for n, qid in enumerate(selected, 1):
        item = by_qid[qid]
        records = item['records']
        if set(records) != set('ABC'):
            raise ValueError('Every selected item needs all three retrieval conditions')
        if item['dataset'] == 'detectiveqa':
            q = dev[qid]
            paragraphs = prefix_for_question(load(ROOT / f'data/processed/books/{q["book_id"]}.json'), q['cutoff_exclusive'])
            group = q['book_id']
        else:
            q = extra[qid]
            paragraphs = q['paragraphs']
            group = q['group_id']
        by_pid = {p['pid']: p['text'] for p in paragraphs}
        order = sorted('ABC', key=lambda condition: hashlib.sha256((SEED + qid + ':' + condition).encode()).hexdigest())
        variants = []
        for label, condition in zip('XYZ', order):
            record = records[condition]
            if record['question'] != q['question'] or record['options'] != q['options']:
                raise ValueError('Prepared question does not match retrieved record')
            # Verify every visible excerpt stays within supplied safe input, without reading gold.
            for chunk in record['evidence']:
                for s in chunk['spans']:
                    if s['pid'] not in by_pid or not 0 <= s['start'] < s['end'] <= len(by_pid[s['pid']]):
                        raise ValueError('Evidence span outside safe context')
            variants.append({'label': label, 'chunks': record['evidence']})
        cases.append({'number': n, 'qid': qid, 'dataset': item['dataset'], 'group_id': group,
                      'question': q['question'], 'options': q['options'], 'paragraphs': paragraphs, 'variants': variants})
        keys.append({'qid': qid, 'condition_by_anonymous_label': dict(zip('XYZ', order))})
    out.mkdir()
    metadata = [{'case': c['number'], 'qid': c['qid'], 'dataset': c['dataset'], 'group_id': c['group_id']} for c in cases]
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'purpose': 'local development human review; ratings not completed',
                'selection': 'First four question IDs per dataset by SHA256(human-review-20261111:+qid); no metric or label filtering',
                'anonymous_condition_order': 'SHA256(seed+qid+condition) then X/Y/Z; do not open condition_key until independent reviews are saved',
                'sample_ids': selected, 'counts': {'detectiveqa': 4, 'musique': 4, 'quality': 4},
                'cases': metadata, 'reference_answers_loaded': False, 'source_text_in_exports': False,
                'input_files': [{'file': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in record_paths],
                'content_handling': 'HTML contains source excerpts and safe context; local group review only, never commit to public GitHub.'}
    write(out / 'manifest.json', manifest)
    write(out / 'condition_key.json', keys)
    template = {'schema_version': 1, 'packet_manifest_sha256': digest(out / 'manifest.json'), 'reviewer': '', 'exported_utc': None,
                'independent_review_confirmed': False, 'status': 'not_started',
                'cases': [{'qid': c['qid'], 'dataset': c['dataset'], 'group_id': c['group_id'],
                           'answerable_from_full_context': None, 'question_or_options_ambiguous': None, 'case_notes': '',
                           'variants': [{'anonymous_label': v['label'], 'evidence_sufficient': None, 'evidence_notes': ''} for v in c['variants']]} for c in cases]}
    write(out / 'empty_review_template.json', template)
    parts = []
    names = {'detectiveqa': 'DetectiveQA 小说', 'musique': 'MuSiQue 多材料', 'quality': 'QuALITY 长文章'}
    for c in cases:
        cid = f'case{c["number"]}'
        options = ''.join(f'<li><strong>{html.escape(k)}</strong> {html.escape(v)}</li>' for k, v in (c['options'] or {}).items())
        blocks = []
        for variant in c['variants']:
            field = cid + '_' + variant['label']
            excerpts = ''.join('<pre>' + html.escape(chunk['text']) + '</pre>' for chunk in variant['chunks'])
            blocks.append(f'<section class="variant"><h3>材料组 {variant["label"]}</h3>{excerpts}'
                          + checkbox_select(field + '_enough', '仅凭这一组材料，是否足以支持一个可靠答案？')
                          + f'<label for="{field}_notes">支撑或缺失了什么？请写具体段号；不确定可说明原因。</label><textarea id="{field}_notes" data-review-field></textarea></section>')
        source = ''.join(f'[{p["pid"]}] {p["text"]}\n\n' for p in c['paragraphs'])
        parts.append(f'<article id="{cid}"><h2>{c["number"]:02} · {names[c["dataset"]]}</h2>'
                     f'<p class="id">{html.escape(c["qid"])}</p><p class="question">{html.escape(c["question"])}</p><ul>{options}</ul>'
                     '<details><summary>展开本题完整可用材料（小说只含答案揭晓前的安全前缀）</summary>'
                     f'<pre>{html.escape(source)}</pre></details>'
                     + checkbox_select(cid + '_answerable', '阅读可用材料后，这道题是否可以有根据地作答？')
                     + checkbox_select(cid + '_ambiguous', '题目或选项是否存在歧义、多个可支持的答案？')
                     + f'<label for="{cid}_notes">题目层面的备注（例如实体混接、缺少证言）。</label><textarea id="{cid}_notes" data-review-field></textarea>'
                     + ''.join(blocks) + '</article>')
    data = json.dumps(template, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    nav = ''.join(f'<a href="#case{c["number"]}">{c["number"]:02}</a>' for c in cases)
    document = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>开发集证据复核 · 12题</title>
<style>body{max-width:980px;margin:28px auto;padding:0 22px 80px;background:#f7f8f5;color:#18312c;font:17px/1.8 "Microsoft YaHei",sans-serif}h1{font-size:30px}h2{border-bottom:2px solid #ccd9d0}article{margin:36px 0;padding:24px;background:white;border:1px solid #d4dfd9;border-radius:12px}label{display:block;margin-top:18px;font-weight:600}select,input,textarea,button{font:inherit;border:1px solid #94aaa0;border-radius:6px;padding:9px;max-width:100%;box-sizing:border-box}select{width:100%;background:white}textarea{width:100%;min-height:95px;background:#fcfdfb}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.9 "Microsoft YaHei",sans-serif;background:#f1f4ef;padding:14px;border-radius:7px}details{margin:18px 0}summary{cursor:pointer;font-weight:600}.variant{border-top:1px dashed #a9bdb0;margin-top:28px;padding-top:8px}.id{font-size:13px;color:#566e64}.question{font-size:19px;font-weight:bold}nav{display:flex;gap:10px;flex-wrap:wrap}nav a{padding:4px 12px;background:#e5eee6;border-radius:6px;color:#15513a}.toolbar{position:sticky;top:0;background:#e9f1e9;border:1px solid #cbdccc;padding:12px;z-index:2}.toolbar button{background:#1c5c43;color:white;cursor:pointer}.note{padding:16px;background:#fff4d5}.instructions{font-size:16px}#message{font-size:14px}footer{font-size:14px;color:#51685a}</style></head><body>
<h1>先看证据，再讨论结论</h1><p>12 道开发题 · 3 种材料 · 未完成人工评分</p>
<div class="instructions"><p>每题由两人分别复核，先导出各自意见，再由第三人处理分歧并轮换角色。开始前不要阅读其他人的评分、参考答案、实验成绩或 condition_key.json。</p>
<p>每题有 X、Y、Z 三组检索材料，排序在不同题中可能不同；字母不表示好坏，也不固定对应某种方法。先看题目，再判断每组是否提供了足够证据；需要核对时展开完整可用材料。没有实际阅读的部分请留空或选“无法判断”。不必猜标准答案。</p>
<p>本轮尚未生成模型回答，因此这里不评价模型答案或解释。多材料题问到的实体关系是否连得上、文章题是否有多个合理选项，都可以写进备注。空白不能当作“没有问题”。</p></div>
<div class="note">本页包含原文，仅供本地小组核验，勿上传公开 GitHub。离开页面前请导出；浏览器不会自动保存。本页没有网络请求。</div>
<label for="reviewer">你的姓名或可辨认的组内代号（自行填写）</label><input id="reviewer" data-review-field autocomplete="off" placeholder="尚未填写">
<label><input type="checkbox" id="independent" data-review-field> 我确认这些意见由我独立填写，尚未参考其他组员的评分。</label>
<div class="toolbar"><button type="button" id="download">导出我的评分 JSON</button> <label style="display:inline;font-weight:normal">恢复已导出的记录 <input type="file" id="restore" accept="application/json,.json"></label><div id="message" role="status">尚未填写，不会自动生成评分。</div></div>
<nav>''' + nav + '</nav>' + ''.join(parts) + '''<footer>没有参考答案、支持段标记或方法身份。导出文件仅包含实际填写的意见、题号与匿名材料组，不包含原文。缺失判断始终保留为空。</footer>
<script id="review-template" type="application/json">''' + data + '''</script>
<script>
"use strict";
const template=JSON.parse(document.getElementById('review-template').textContent);
const val=id=>document.getElementById(id).value;
const tri=id=>val(id)||null;
function collect(){
 const r=JSON.parse(JSON.stringify(template));
 r.reviewer=val('reviewer').trim();r.independent_review_confirmed=document.getElementById('independent').checked;
 r.exported_utc=new Date().toISOString();let done=0;
 r.cases.forEach((c,i)=>{const p='case'+(i+1);c.answerable_from_full_context=tri(p+'_answerable');c.question_or_options_ambiguous=tri(p+'_ambiguous');c.case_notes=val(p+'_notes');
   c.variants.forEach(v=>{const x=p+'_'+v.anonymous_label;v.evidence_sufficient=tri(x+'_enough');v.evidence_notes=val(x+'_notes');});
   if(c.answerable_from_full_context&&c.question_or_options_ambiguous&&c.variants.every(v=>v.evidence_sufficient))done++;
 });
 r.status=done===r.cases.length&&r.reviewer&&r.independent_review_confirmed?'independent_review_complete':'in_progress';
 return [r,done];
}
function progress(){const [r,n]=collect();document.getElementById('message').textContent=`已填写全部判断的题目：${n}/${r.cases.length}。状态：${r.status==='independent_review_complete'?'独立复核已填写完整':'仍有未填写项，可随时导出进度'}。`;}
document.querySelectorAll('[data-review-field]').forEach(x=>x.addEventListener('input',progress));
document.getElementById('download').addEventListener('click',()=>{
 const [r]=collect();const blob=new Blob([JSON.stringify(r,null,2)],{type:'application/json;charset=utf-8'});const u=URL.createObjectURL(blob);const a=document.createElement('a');a.href=u;a.download='evidence_review_'+new Date().toISOString().replace(/[:.]/g,'-')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);progress();
});
document.getElementById('restore').addEventListener('change',async e=>{
 try{const file=e.target.files[0];if(!file)return;const r=JSON.parse(await file.text());
 if(r.packet_manifest_sha256!==template.packet_manifest_sha256||r.cases.length!==template.cases.length)throw new Error('记录不属于本轮复核包');
 const allowed=new Set([null,'yes','no','unclear']);
 r.cases.forEach((c,i)=>{if(c.qid!==template.cases[i].qid||!allowed.has(c.answerable_from_full_context)||!allowed.has(c.question_or_options_ambiguous)||typeof c.case_notes!=='string'||c.variants.length!==3)throw new Error('题号或字段不匹配');
 c.variants.forEach((v,j)=>{if(v.anonymous_label!==template.cases[i].variants[j].anonymous_label||!allowed.has(v.evidence_sufficient)||typeof v.evidence_notes!=='string')throw new Error('材料组字段不匹配');});});
 if(typeof r.reviewer!=='string'||typeof r.independent_review_confirmed!=='boolean')throw new Error('复核人字段不匹配');
 document.getElementById('reviewer').value=r.reviewer;document.getElementById('independent').checked=r.independent_review_confirmed;
 r.cases.forEach((c,i)=>{const p='case'+(i+1);document.getElementById(p+'_answerable').value=c.answerable_from_full_context||'';document.getElementById(p+'_ambiguous').value=c.question_or_options_ambiguous||'';document.getElementById(p+'_notes').value=c.case_notes;
 c.variants.forEach(v=>{const x=p+'_'+v.anonymous_label;document.getElementById(x+'_enough').value=v.evidence_sufficient||'';document.getElementById(x+'_notes').value=v.evidence_notes;});});progress();
 }catch(err){document.getElementById('message').textContent='无法恢复：'+err.message;}
});
</script></body></html>'''
    (out / 'evidence_review.html').write_text(document, encoding='utf8')
    (out / '使用说明.txt').write_text('打开 evidence_review.html。12道开发题，每种数据4道；全部人工字段为空。\n每题两人先独立评估，分别导出JSON，再由第三人处理分歧并轮换角色。\n本轮只有检索材料，没有模型回答，不评模型解释。没读完或无法确认的部分留空/无法判断。\n先保存独立评分，再查看 condition_key.json 对照方法；不要让方法名称影响判断。\n导出文件不含原文。HTML含原文，只供本地组内核验，勿上传公开仓库。\n本页不联网、不自动保存；退出前导出，后续可恢复同一包的JSON。\n', encoding='utf8')
    return out, manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--detective-run', default='dev_retrieval_20261111_01')
    ap.add_argument('--cross-run', default='cross_dataset_dev_20261111_01')
    args = ap.parse_args()
    for value in (args.detective_run, args.cross_run):
        if Path(value).name != value or value in ('.', '..'):
            raise ValueError('Run names must be single directory names')
    dest, manifest = make_packet(ROOT / 'runs' / args.detective_run, ROOT / 'runs' / args.cross_run)
    print(json.dumps({'review_packet': str(dest), 'cases': len(manifest['sample_ids']), 'human_ratings': 'all blank'}, ensure_ascii=False))
