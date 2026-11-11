"""Offline Chromium rendering of the generated artifact; no websites are opened."""
from pathlib import Path
import json
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/qa'
CHROME=Path(r'C:\Program Files\Google\Chrome\Application\chrome.exe')
source=(ROOT/'output/html/fundo_profile_report_slides.html').read_text(encoding='utf-8')
driver=r"""
<script>
window.addEventListener('load',function(){setTimeout(async function(){
 const pause=ms=>new Promise(r=>setTimeout(r,ms));
 const slides=[...document.querySelectorAll('.slide')], result=[];
 for(let i=0;i<slides.length;i++){
   location.hash='slide-'+(i+1); await pause(750);
   const s=slides[i], body=s.querySelector('.s-body');
   let crops=[];
   s.querySelectorAll('.side-card,.finding-card,.kpi-card,.risk-card,.note-strip,.table-box').forEach(c=>{
     if(c.scrollHeight>c.clientHeight+3 || c.scrollWidth>c.clientWidth+3) crops.push({class:c.className,h:c.clientHeight,scroll:c.scrollHeight,w:c.clientWidth,scrollW:c.scrollWidth});
   });
   let plots=[...s.querySelectorAll('.js-plotly-plot')].map(p=>({w:p.clientWidth,h:p.clientHeight,traces:(p.data||[]).length,box:p.closest('.chart-box').clientHeight}));
   let escaped=[];
   if(body){const bounds=body.getBoundingClientRect();[...body.children].forEach(c=>{const b=c.getBoundingClientRect();if(b.bottom>bounds.bottom+2 || b.right>bounds.right+2)escaped.push(c.className);});}
   result.push({slide:i+1,title:s.getAttribute('aria-label'),body:body?{height:body.clientHeight,scroll:body.scrollHeight}:null,crops,escaped,plots});
 }
 const pre=document.createElement('pre');pre.id='FUNDO_QA';pre.textContent=JSON.stringify({errors:window.renderErrors||[],slides:result});document.body.appendChild(pre);
},500)});
</script>
"""
hook="<style>*,*::before,*::after{animation:none!important;transition:none!important}.reveal{opacity:1!important;transform:none!important}</style><script>window.renderErrors=[];window.addEventListener('error',e=>renderErrors.push(e.message));window.addEventListener('unhandledrejection',e=>renderErrors.push(String(e.reason)));</script>"
check=OUT/'fundo_layout_check.html'
check.write_text(source.replace('<head>','<head>'+hook).replace('</body>',driver+'</body>'),encoding='utf-8')
common=[str(CHROME),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--allow-file-access-from-files','--user-data-dir='+str(OUT/'chrome-offline-review'),'--window-size=1700,1020']
if '--screenshots' not in sys.argv:
    proc=subprocess.run(common+['--virtual-time-budget=25000','--dump-dom',check.as_uri()],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=55)
    match=re.search(r'<pre id="FUNDO_QA">(.*?)</pre>',proc.stdout,re.S)
    if not match:
        print(proc.stderr[:4000]);raise SystemExit('Chromium did not return layout evidence')
    import html
    result=json.loads(html.unescape(match.group(1)))
    (OUT/'fundo_layout_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for row in result['slides']:print(json.dumps(row,ensure_ascii=False))
    print('SCRIPT ERRORS',result['errors'])
else:
    still=OUT/'fundo_still_preview.html'
    still.write_text(source.replace('<head>','<head>'+hook),encoding='utf-8')
    for index in [2,3,6,7,10,14,16,17,20,24,25]:
        target=OUT/f'fundo_slide_{index:02d}.png'
        proc=subprocess.run(common+['--virtual-time-budget=1600','--screenshot='+str(target),still.as_uri()+f'#slide-{index}'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=40)
        print(index,target.exists(),target.stat().st_size if target.exists() else proc.stderr[:150])
