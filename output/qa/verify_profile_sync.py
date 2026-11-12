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
    'swap_in':'existing_swap_in_profile_report_slides.html',
    'swap_out':'existing_swap_out_profile_report_slides.html',
    'compare':'compare_profile_report_slides.html',
    'compare_swap':'compare_swap_profile_report_slides.html',
    'raw_swap':'swap_in_out_profile_report_slides.html',
}
SAMPLE_TITLES={
    '收入水平接近，差异在来源结构：WageGo 更靠工资，Fundo 更靠福利',
    'WageGo 的借贷依赖、还款负担与拒付暴露都显著更高',
    '支出规模接近，WageGo 固定账单负担更低、赌博与手续费更高',
    '负盈余与薄缓冲是共同底盘，WageGo 的账户余额更薄',
    '收入几乎相同，差距在负债与履约','收入只差 2%，WageGo 的还款负担却接近 Fundo 的两倍',
    '借贷依赖与拒付暴露，是两品牌最大分化','薄流动性是共同底盘，WageGo 的赌博与手续费更高',
    '共同月份校准后，品牌差异仍成立','换成工资+福利分母，负担明显上升',
    '多头覆盖差异大，同组内拒付仍更高','近期变化接近，双重压力组未更高拒付',
    '快照为正，仍可能隐藏历史压力','WageGo 净流出更高，但同机构数组内无一致梯度',
    '重压型拒付最高，盈余型并非最低','主要差异集中在负担、多头与履约','收入来源与偿债承受力','消费行为：看净流出，并控制多头暴露',
    '近期压力：收入回落与还款上升叠加','变量审查：补充关系，保留边界',
    '就业形态：行业与发薪周期','就业形态：发薪周期与偿债负担','WageGo 雇主覆盖更广，行业码不可比',
}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',choices=list(REPORTS)+['all'],default='all')
    parser.add_argument('--screenshots',action='store_true')
    parser.add_argument('--all-slides',action='store_true')
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
                if not args.all_slides and slide.get('aria-label') not in SAMPLE_TITLES:
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
