#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Headless layout diagnostics for generated slide decks (dev helper).

Walks every slide (clicking the next button), then reports body overflow,
inner cropping and chart box vs. plot size for the active slide.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
CHECKER = r"""
<script>
window.addEventListener('load', function(){
  setTimeout(async function(){
   var sleep=function(ms){return new Promise(function(r){setTimeout(r,ms)})};
   var out=[];
   var next=document.getElementById('nextBtn');
   var slides=Array.prototype.slice.call(document.querySelectorAll('.slide'));
   for(var i=0;i<slides.length;i++){
     if(!slides[i].classList.contains('is-active')){ next.click(); await sleep(700); }
     var s=slides[i];
     var label=s.getAttribute('aria-label');
     var body=s.querySelector('.s-body');
     var info='SLIDE '+i+' ['+label+']';
     if(body){
       info+=' body='+body.clientHeight+' scroll='+body.scrollHeight;
       Array.prototype.forEach.call(body.children,function(c){
         if(c.scrollHeight>c.clientHeight+2 && getComputedStyle(c).overflow!=='visible'){
           info+=' CROP:'+(c.className||c.tagName)+' '+c.clientHeight+'/'+c.scrollHeight;
         }
       });
     }
     s.querySelectorAll('.js-plotly-plot').forEach(function(p){
       var box=p.closest('.chart-box');
       info+=' | plot '+p.offsetWidth+'x'+p.offsetHeight+' box '+(box?box.clientWidth+'x'+box.clientHeight:'?');
     });
     out.push(info);
   }
   var pre=document.createElement('pre'); pre.id='DECK_DIAG'; pre.textContent=out.join('\n');
   document.body.appendChild(pre);
  }, 900);
});
</script>
"""


def check(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    tmp = Path(tempfile.gettempdir()) / ("_deck_diag_" + path.stem + ".html")
    tmp.write_text(src.replace("</body>", CHECKER + "</body>"), encoding="utf-8")
    url = tmp.resolve().as_uri()
    out = subprocess.run(
        [
            str(CHROME),
            "--headless=new",
            "--disable-gpu",
            "--virtual-time-budget=30000",
            "--window-size=1700,1020",
            "--dump-dom",
            url,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout
    m = re.search(r'<pre id="DECK_DIAG">(.*?)</pre>', out, re.S)
    return m.group(1) if m else "NO DIAG (page failed to render)"


def main() -> None:
    for arg in sys.argv[1:]:
        path = Path(arg)
        print("=" * 20, path.name)
        print(check(path))


if __name__ == "__main__":
    main()
