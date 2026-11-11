"""Offline rendering and screenshots for both profiles and their comparison."""
from pathlib import Path
import argparse
import ast
import hashlib
import html
import json
import re
import subprocess

from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/qa'
REPORTS={
    'fundo':'fundo_profile_report_slides.html',
    'wagego':'wagego_profile_report_slides.html',
    'compare':'compare_profile_report_slides.html',
}
SAMPLE_TITLES={
    '先校准口径，再比较品牌差异','收入分母统一后，偿债负担如何变化',
    '多头还款：区分关系数量与组内表现','近期变化：用互斥窗口检验资金压力',
    '流动性：快照之外的历史压力','行为净流出：在同样借贷关系下比较',
    '分层指标明细','收入来源与偿债承受力','消费行为：看净流出，并控制多头暴露',
    '近期压力：收入回落与还款上升叠加','变量审查：补充关系，保留边界',
    '就业形态：行业与发薪周期','就业形态：发薪周期与偿债负担','就业形态：覆盖率与发薪周期',
}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',choices=list(REPORTS)+['all'],default='all')
    parser.add_argument('--screenshots',action='store_true')
    args=parser.parse_args()
    # Reuse the established settled-layout checker without executing that script.
    nodes=ast.parse((OUT/'verify_fundo_layout.py').read_text(encoding='utf-8'))
    snippets={n.targets[0].id:ast.literal_eval(n.value) for n in nodes.body
              if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
              and n.targets[0].id in {'driver','hook'}}
    common=[r'C:\Program Files\Google\Chrome\Application\chrome.exe','--headless=new',
        '--disable-gpu','--no-first-run','--no-default-browser-check',
        '--allow-file-access-from-files','--user-data-dir='+str(OUT/'chrome-sync-review'),
        '--window-size=1700,1020']
    for key in REPORTS if args.report=='all' else [args.report]:
        source=ROOT/'output/html'/REPORTS[key]
        source_bytes=source.read_bytes()
        raw=source_bytes.decode('utf-8').replace('\r\n','\n')
        page=BeautifulSoup(raw,'html.parser')
        slides=page.select('.slide')
        checked=raw.replace('<head>','<head>'+snippets['hook'])
        if args.screenshots:
            still=OUT/f'{key}_sync_still.html'
            still.write_text(checked,encoding='utf-8')
            for i,slide in enumerate(slides,1):
                if slide.get('aria-label') not in SAMPLE_TITLES:
                    continue
                target=OUT/f'{key}_sync_slide_{i:02d}.png'
                result=subprocess.run(common+['--virtual-time-budget=1600','--screenshot='+str(target),still.as_uri()+f'#slide-{i}'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=40)
                if not target.exists():
                    raise RuntimeError(result.stderr[:2000])
                print(key,i,slide.get('aria-label'),target.stat().st_size,flush=True)
        else:
            temp=OUT/f'{key}_sync_layout.html'
            temp.write_text(checked.replace('</body>',snippets['driver']+'</body>'),encoding='utf-8')
            budget=max(25000,1000+900*len(slides))
            result=subprocess.run(common+['--virtual-time-budget='+str(budget),'--dump-dom',temp.as_uri()],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=55)
            match=re.search(r'<pre id="FUNDO_QA">(.*?)</pre>',result.stdout,re.S)
            if not match:
                raise RuntimeError('No layout evidence: '+result.stderr[:2500])
            info=json.loads(html.unescape(match.group(1)))
            info.update(report=REPORTS[key],sha256=hashlib.sha256(source_bytes).hexdigest())
            (OUT/f'{key}_sync_layout_results.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
            issues=[s for s in info['slides'] if s['crops'] or s['escaped'] or (s['body'] and s['body']['scroll']>s['body']['height']+2)]
            print(key,'slides',len(info['slides']),'charts',sum(len(s['plots']) for s in info['slides']),'issues',json.dumps(issues,ensure_ascii=False),'errors',info['errors'],flush=True)


if __name__=='__main__':
    main()
