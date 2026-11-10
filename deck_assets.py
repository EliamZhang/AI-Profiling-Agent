#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Slide-deck theme extracted from bscat_fiskil_slides.html (layout sample)."""

CSS = r""":root{
    --navy:#0B2B4A;
    --navy-2:#173F63;
    --blue:#2C6CB0;
    --blue-soft:#EDF4FB;
    --ink:#25384B;
    --text:#3A4A5C;
    --muted:#66788A;
    --faint:#8D9AA8;
    --line:#DFE7EF;
    --line-soft:#EDF2F7;
    --soft:#F5F8FB;
    --white:#FFFFFF;
    --green:#0E7558;
    --green-bg:#EAF7F1;
    --green-line:#CFE8DD;
    --amber:#9A5B13;
    --amber-bg:#FDF4E6;
    --amber-line:#F0DFC4;
    --chart-navy:#1D4E79;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{width:100%;height:100%;overflow:hidden}
  body{
    font-family:"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC",sans-serif;
    color:var(--text);background:#DCE3EB;
    -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;
  }
  ::selection{background:#DCE9F8;color:var(--navy)}
  button{font:inherit}
  .num,.big,.kpi-value,.exec-stat b,.case-val,.hc-val,.metric b,.meta b{font-variant-numeric:tabular-nums}

  /* ===== Stage ===== */
  .stage-wrap{
    position:fixed;inset:0;display:flex;align-items:center;justify-content:center;overflow:hidden;
    padding:40px 40px 76px;
    background:radial-gradient(1200px 620px at 50% 0%,#EEF3F8 0%,#DDE4EC 58%,#D4DDE7 100%);
  }
  .stage{
    position:relative;width:1600px;height:900px;flex:none;
    background:#fff;border-radius:14px;overflow:hidden;
    box-shadow:0 42px 110px rgba(11,43,74,.24),0 1px 0 rgba(255,255,255,.8) inset;
    transform-origin:center center;
  }
  .slide{
    position:absolute;inset:0;padding:52px 78px 46px;
    display:flex;flex-direction:column;background:#fff;
    opacity:0;visibility:hidden;pointer-events:none;transform:translateY(14px);
    transition:opacity .38s ease,transform .38s ease,visibility .38s;
  }
  .slide.is-active{opacity:1;visibility:visible;pointer-events:auto;transform:none;z-index:2}
  .slide.cover-slide,.slide.divider,.slide.closing{color:#fff}

  /* ===== Shared type ===== */
  .s-top{display:flex;align-items:center;justify-content:space-between;gap:24px;flex:none;margin-bottom:24px}
  .s-brand{font-size:12px;font-weight:700;letter-spacing:.16em;color:var(--navy);white-space:nowrap}
  .s-brand em{font-style:normal;color:var(--blue);margin:0 2px}
  .s-section{font-size:10.5px;font-weight:700;letter-spacing:.16em;color:var(--faint);text-transform:uppercase;white-space:nowrap}
  .s-head{flex:none;margin-bottom:26px}
  .s-kicker{font-size:11.5px;font-weight:700;letter-spacing:.22em;color:var(--blue);text-transform:uppercase;margin-bottom:10px}
  .s-head h2{font-size:42px;line-height:1.18;letter-spacing:-.015em;color:var(--navy);font-weight:700}
  .s-sub{margin-top:9px;font-size:16px;line-height:1.65;color:var(--muted);max-width:1060px}
  .s-body{flex:1;min-height:0;display:flex;flex-direction:column;gap:18px}
  .s-body.center{justify-content:center}

  .status{
    display:inline-flex;align-items:center;gap:7px;padding:7px 13px;border-radius:999px;
    font-size:12.5px;font-weight:700;line-height:1.25;border:1px solid transparent;white-space:nowrap;
  }
  .status i{width:6px;height:6px;border-radius:50%;background:currentColor;flex:none}
  .status.green{background:var(--green-bg);color:var(--green);border-color:var(--green-line)}
  .status.amber{background:var(--amber-bg);color:var(--amber);border-color:var(--amber-line)}
  .status.blue{background:var(--blue-soft);color:var(--blue);border-color:#D5E4F3}

  .reveal{opacity:0;transform:translateY(15px)}
  .slide.is-active .reveal{animation:rise .58s cubic-bezier(.22,.61,.36,1) forwards}
  .slide.is-active .d1{animation-delay:.07s}
  .slide.is-active .d2{animation-delay:.14s}
  .slide.is-active .d3{animation-delay:.21s}
  .slide.is-active .d4{animation-delay:.28s}
  @keyframes rise{to{opacity:1;transform:none}}

  .fill{width:0}
  .slide.is-active .fill{animation:grow 1.05s cubic-bezier(.22,.61,.36,1) forwards}
  @keyframes grow{to{width:var(--w)}}

  /* ===== Cover ===== */
  .cover-slide{
    padding:54px 80px 44px;justify-content:space-between;
    background:
      radial-gradient(1050px 620px at 82% -8%,#205074 0%,rgba(32,80,116,0) 62%),
      linear-gradient(135deg,#0B2B4A 0%,#0A2440 100%);
  }
  .cover-slide::before{
    content:"";position:absolute;right:-100px;top:-90px;width:650px;height:650px;
    background-image:radial-gradient(rgba(255,255,255,.16) 1px,transparent 1px);
    background-size:24px 24px;opacity:.4;pointer-events:none;
    -webkit-mask-image:radial-gradient(circle at center,#000 0%,transparent 68%);
    mask-image:radial-gradient(circle at center,#000 0%,transparent 68%);
  }
  .cover-top{display:flex;align-items:center;justify-content:space-between;gap:20px;position:relative;z-index:1}
  .cover-brand{font-size:13px;font-weight:700;letter-spacing:.2em;color:#fff}
  .cover-brand em{font-style:normal;color:#8FB8DC;margin:0 2px}
  .cover-tag{font-size:10px;letter-spacing:.2em;color:rgba(255,255,255,.52);font-weight:700}
  .cover-main{position:relative;z-index:1;max-width:1230px}
  .cover-eyebrow{display:flex;align-items:center;gap:14px;font-size:11.5px;letter-spacing:.2em;color:#9FC4E2;font-weight:700;text-transform:uppercase}
  .cover-eyebrow::before{content:"";width:30px;height:1px;background:#6FA4CB}
  .cover-main h1{font-size:64px;line-height:1.14;letter-spacing:-.025em;color:#fff;font-weight:700;margin:26px 0 0}
  .cover-main p{font-size:18px;line-height:1.8;color:rgba(255,255,255,.68);max-width:1030px;margin-top:24px}
  .cover-status{display:flex;gap:12px;flex-wrap:wrap;margin-top:32px}
  .cover-status .status{padding:9px 15px;font-size:12.5px}
  .cover-status .status.green{background:rgba(29,147,111,.15);border-color:rgba(126,209,181,.32);color:#A8E7CE}
  .cover-status .status.amber{background:rgba(201,138,46,.14);border-color:rgba(240,194,126,.3);color:#F3CE95}
  .cover-bottom{
    display:flex;align-items:flex-end;justify-content:space-between;gap:30px;
    padding-top:24px;border-top:1px solid rgba(255,255,255,.14);position:relative;z-index:1;
  }
  .cover-metrics{display:flex;gap:58px;flex-wrap:wrap}
  .metric .k{font-size:9.5px;letter-spacing:.16em;color:rgba(255,255,255,.45);font-weight:700;text-transform:uppercase}
  .metric b{display:block;font-size:25px;line-height:1.2;color:#fff;margin-top:5px;font-weight:700}
  .metric span{display:block;font-size:11.5px;color:rgba(255,255,255,.5);margin-top:2px}
  .cover-hint{font-size:11px;color:rgba(255,255,255,.45);white-space:nowrap}
  .cover-hint b{color:rgba(255,255,255,.75);font-weight:600}

  /* ===== Divider ===== */
  .divider{
    padding:70px 90px 58px;justify-content:space-between;overflow:hidden;
    background:
      radial-gradient(900px 540px at 84% 2%,#245A83 0%,rgba(36,90,131,0) 64%),
      linear-gradient(135deg,#0B2B4A 0%,#0A2440 100%);
  }
  .divider::after{
    content:attr(data-num);position:absolute;right:64px;bottom:-28px;
    font-size:260px;line-height:1;font-weight:700;letter-spacing:-.07em;
    color:rgba(255,255,255,.055);pointer-events:none;
  }
  .divider .s-kicker{color:#9FC4E2}
  .divider h2{font-size:64px;line-height:1.1;letter-spacing:-.025em;color:#fff;font-weight:700;margin-top:14px}
  .divider p{max-width:760px;margin-top:20px;font-size:19px;line-height:1.75;color:rgba(255,255,255,.68)}
  .divider .status{margin-top:28px}
  .divider .status.green{background:rgba(29,147,111,.15);border-color:rgba(126,209,181,.32);color:#A8E7CE}
  .divider .status.amber{background:rgba(201,138,46,.14);border-color:rgba(240,194,126,.3);color:#F3CE95}
  .divider-meta{display:flex;gap:64px;padding-top:24px;border-top:1px solid rgba(255,255,255,.14);position:relative;z-index:1;flex-wrap:wrap}
  .divider-meta b{display:block;font-size:34px;line-height:1.1;color:#fff;font-weight:700}
  .divider-meta span{display:block;font-size:11px;letter-spacing:.08em;color:rgba(255,255,255,.48);margin-top:5px}

  /* ===== Cards / common ===== */
  .exec-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;flex:1;min-height:0}
  .exec-card{
    display:flex;flex-direction:column;background:#fff;border:1px solid var(--line);
    border-radius:18px;padding:26px 28px;position:relative;overflow:hidden;
    box-shadow:0 12px 34px rgba(11,43,74,.045);
  }
  .exec-card::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px}
  .exec-card.green::before{background:var(--green)}
  .exec-card.amber::before{background:#C98A2E}
  .exec-card-top{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap}
  .exec-main{flex:1;min-height:0;display:flex;flex-direction:column;justify-content:center}
  .exec-label{font-size:11px;letter-spacing:.12em;color:var(--muted);font-weight:700;text-transform:uppercase}
  .exec-card h3{font-size:25px;line-height:1.38;color:var(--navy);font-weight:700;margin:16px 0 10px;letter-spacing:-.01em}
  .exec-card p{font-size:15px;line-height:1.7;color:var(--text)}
  .exec-stats{display:flex;gap:30px;flex-wrap:wrap;margin-top:24px;padding-top:18px;border-top:1px solid var(--line-soft)}
  .exec-stat b{display:block;font-size:29px;line-height:1.15;color:var(--navy);font-weight:700}
  .exec-stat span{display:block;font-size:11px;color:var(--faint);margin-top:3px}
  .caveat{
    flex:none;display:flex;align-items:flex-start;gap:13px;padding:15px 20px;
    background:#FFFBF3;border:1px solid #F1E2C9;border-left:4px solid #C98A2E;border-radius:12px;
  }
  .caveat .mark{
    width:21px;height:21px;border-radius:50%;background:#F2E0C4;color:#8A5A13;
    font-size:12px;font-weight:800;display:grid;place-items:center;flex:none;margin-top:1px;
  }
  .caveat b{display:block;font-size:12.5px;color:#8A5A13;letter-spacing:.02em}
  .caveat p{font-size:12.5px;line-height:1.65;color:#7A5A2A;margin-top:2px}

  /* ===== KPI ===== */
  .kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;flex:none}
  .kpi-card{background:#FBFCFE;border:1px solid var(--line);border-radius:16px;padding:22px 24px 20px}
  .kpi-label{font-size:11px;letter-spacing:.07em;color:var(--muted);font-weight:700}
  .kpi-value{font-size:43px;line-height:1.15;color:var(--navy);font-weight:700;letter-spacing:-.02em;margin-top:8px}
  .kpi-value small{font-size:15px;color:var(--muted);font-weight:500;letter-spacing:0;margin-left:4px}
  .kpi-note{font-size:11.5px;line-height:1.5;color:var(--faint);margin-top:2px}
  .split{display:grid;grid-template-columns:1.08fr .92fr;gap:18px;flex:1;min-height:0}
  .statement{
    background:linear-gradient(135deg,#0B2B4A 0%,#173F63 100%);color:#fff;border-radius:18px;
    padding:32px 34px;display:flex;flex-direction:column;justify-content:center;position:relative;overflow:hidden;
  }
  .statement::after{
    content:"";position:absolute;right:-40px;bottom:-50px;width:260px;height:260px;border-radius:50%;
    border:1px solid rgba(255,255,255,.1);
  }
  .statement .status{align-self:flex-start;background:rgba(29,147,111,.18);border-color:rgba(126,209,181,.32);color:#A8E7CE}
  .statement h3{font-size:31px;line-height:1.35;color:#fff;font-weight:700;margin:18px 0 10px;letter-spacing:-.01em;position:relative;z-index:1}
  .statement p{font-size:15px;line-height:1.8;color:rgba(255,255,255,.7);max-width:620px;position:relative;z-index:1}
  .statement .mini{display:flex;gap:26px;margin-top:22px;padding-top:18px;border-top:1px solid rgba(255,255,255,.14);position:relative;z-index:1;flex-wrap:wrap}
  .statement .mini b{display:block;font-size:24px;color:#fff;line-height:1.2}
  .statement .mini span{display:block;font-size:10.5px;color:rgba(255,255,255,.48);margin-top:2px}
  .note-card{
    background:#FFFBF3;border:1px solid #F1E2C9;border-left:4px solid #C98A2E;border-radius:16px;
    padding:26px 28px;display:flex;flex-direction:column;justify-content:center;
  }
  .note-card h3{font-size:20px;color:#8A5A13;font-weight:700;margin-bottom:10px}
  .note-card p{font-size:14.5px;line-height:1.8;color:#7A5A2A}
  .note-card .divider-line{height:1px;background:#EEDDBF;margin:16px 0}

  /* ===== Confidence ===== */
  .conf-layout{display:grid;grid-template-columns:390px 1fr;gap:46px;align-items:center;flex:1;min-height:0}
  .donut-wrap{display:flex;flex-direction:column;align-items:center;justify-content:center}
  .donut{
    width:300px;height:300px;border-radius:50%;position:relative;
    background:conic-gradient(from -90deg,#116B5B 0 68.4%,#4C9A8B 68.4% 81.5%,#D3A044 81.5% 95.2%,#B95F5F 95.2% 100%);
    box-shadow:0 18px 40px rgba(11,43,74,.1);
  }
  .donut::after{content:"";position:absolute;inset:42px;background:#fff;border-radius:50%}
  .donut-center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:1;text-align:center}
  .donut-center strong{font-size:40px;line-height:1.15;color:var(--navy);font-weight:700;letter-spacing:-.02em}
  .donut-center span{font-size:11px;color:var(--faint);margin-top:3px}
  .donut-cap{font-size:10.5px;letter-spacing:.12em;color:var(--faint);margin-top:14px;font-weight:600}
  .conf-list{display:flex;flex-direction:column}
  .conf-row{display:grid;grid-template-columns:12px 1fr auto;gap:14px;align-items:center;padding:14px 0;border-bottom:1px dashed var(--line)}
  .conf-row:last-child{border-bottom:none}
  .sw{width:10px;height:10px;border-radius:3px;flex:none}
  .conf-row .name{font-size:16px;color:var(--text);font-weight:600}
  .conf-row .name small{font-size:12px;color:var(--faint);font-weight:400;margin-left:8px}
  .conf-row .pct{font-size:24px;font-weight:700;color:var(--navy);line-height:1.2}
  .conf-note{font-size:12px;line-height:1.65;color:var(--faint);margin-top:12px;padding-top:12px;border-top:1px solid var(--line-soft)}
  .fact-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;flex:none}
  .fact{background:#FBFCFE;border:1px solid var(--line);border-radius:13px;padding:14px 17px}
  .fact b{display:block;font-size:21px;color:var(--navy);line-height:1.2;font-weight:700}
  .fact span{display:block;font-size:10.5px;color:var(--faint);margin-top:3px;line-height:1.4}

  /* ===== Per-case chart ===== */
  .case-panel{
    flex:1;min-height:0;display:flex;flex-direction:column;
    border:1px solid var(--line);border-radius:18px;background:#FBFCFE;padding:20px 26px 16px;
  }
  .case-panel-head{display:flex;align-items:center;justify-content:space-between;gap:16px;flex:none}
  .case-panel-head .title{font-size:14px;font-weight:700;color:var(--navy)}
  .case-legend{display:flex;gap:18px;align-items:center;font-size:10.5px;color:var(--muted);flex-wrap:wrap}
  .case-legend .key{display:inline-flex;align-items:center;gap:6px}
  .case-legend i{width:10px;height:10px;border-radius:2px;display:inline-block}
  .case-plot{flex:1;min-height:0;display:grid;grid-template-columns:repeat(20,1fr);gap:10px;padding:28px 2px 0;align-items:stretch}
  .case-col{height:100%;display:grid;grid-template-rows:1fr 20px;min-width:0}
  .case-bar-area{display:flex;align-items:flex-end;height:100%}
  .case-bar{
    width:100%;height:0;border-radius:4px 4px 0 0;position:relative;
    background:linear-gradient(180deg,#2E6C9F 0%,#1D4E79 100%);
  }
  .case-bar.low{background:linear-gradient(180deg,#E0AE64 0%,#C98A2E 100%)}
  .slide.is-active .case-bar{animation:growH 1s cubic-bezier(.22,.61,.36,1) forwards}
  @keyframes growH{to{height:var(--h)}}
  .case-bar .case-val{
    position:absolute;left:50%;bottom:calc(100% + 5px);transform:translateX(-50%);
    font-size:9.5px;font-weight:700;color:var(--navy);white-space:nowrap;
  }
  .case-bar.low .case-val{color:#9A5B13}
  .case-id{font-size:9px;color:var(--faint);text-align:center;align-self:end;white-space:nowrap;letter-spacing:.01em}
  .case-insights{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;flex:none}
  .insight{background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 17px;display:flex;align-items:baseline;gap:10px}
  .insight b{font-size:23px;line-height:1.2;color:var(--navy);font-weight:700;white-space:nowrap}
  .insight span{font-size:11.5px;line-height:1.45;color:var(--muted)}

  /* ===== Agreement ===== */
  .agree-panel{
    flex:1;min-height:0;display:flex;flex-direction:column;justify-content:center;gap:24px;
    border:1px solid var(--line);border-radius:18px;background:#FBFCFE;padding:28px 32px;
  }
  .agree-head{display:flex;align-items:baseline;justify-content:space-between;gap:16px}
  .agree-head .title{font-size:15px;font-weight:700;color:var(--navy)}
  .agree-head .note{font-size:11.5px;color:var(--faint)}
  .agree-bar{display:flex;height:76px;border-radius:12px;overflow:hidden;background:#EEF2F6}
  .seg{display:flex;align-items:center;justify-content:center;color:#fff;font-size:20px;font-weight:700;flex:0 0 var(--w);min-width:0;overflow:hidden;white-space:nowrap}
  .seg.both{background:linear-gradient(90deg,#173F63,#2E6C9F)}
  .seg.fiskil{background:#A7B7C8;color:#25384B}
  .seg.bscat{background:#5E8BB4}
  .seg.none{background:#E8EDF2;flex:1 1 auto}
  .agree-cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
  .agree-card{background:#fff;border:1px solid var(--line);border-radius:13px;padding:15px 17px}
  .agree-card .k{display:flex;align-items:center;gap:7px;font-size:10.5px;color:var(--muted);font-weight:600}
  .agree-card .k i{width:8px;height:8px;border-radius:2px;display:inline-block;flex:none}
  .agree-card b{display:block;font-size:25px;color:var(--navy);font-weight:700;line-height:1.2;margin-top:7px}
  .agree-card span{display:block;font-size:10.5px;color:var(--faint);margin-top:2px}

  /* ===== High consistency ===== */
  .hc-grid{display:grid;grid-template-columns:1fr 1fr;gap:0 46px;flex:1;align-content:center;min-height:0}
  .hc-col{display:flex;flex-direction:column;justify-content:flex-start}
  .hc-row{display:grid;grid-template-columns:176px minmax(0,1fr) 68px;gap:14px;align-items:center;padding:10.5px 0;border-bottom:1px dashed var(--line)}
  .hc-name{font-size:14.5px;color:var(--navy);font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .hc-track{height:10px;background:#EDF2F7;border-radius:99px;overflow:hidden}
  .hc-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#1D4E79,#3A719F);width:0}
  .slide.is-active .hc-fill{animation:grow 1.05s cubic-bezier(.22,.61,.36,1) forwards}
  .hc-val{font-size:15px;font-weight:700;color:var(--navy);text-align:right}
  .hc-side-note{display:flex;align-items:center;min-height:41px;padding:9px 13px;margin-top:10px;border:1px dashed #D8E2EC;border-radius:10px;background:#F7FAFC;font-size:11.5px;line-height:1.55;color:var(--muted)}
  .hc-note{font-size:11.5px;line-height:1.6;color:var(--faint);flex:none;padding-top:14px;border-top:1px solid var(--line-soft)}

  /* ===== Divergence ===== */
  .div-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;flex:none}
  .div-card{
    background:#fff;border:1px solid var(--line);border-top:4px solid var(--green);
    border-radius:16px;padding:22px 23px;display:flex;flex-direction:column;
  }
  .div-top{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}
  .div-card h3{font-size:20px;line-height:1.35;color:var(--navy);font-weight:700}
  .div-tag{font-size:9.5px;font-weight:700;padding:4px 8px;border-radius:999px;background:var(--green-bg);color:var(--green);white-space:nowrap;border:1px solid var(--green-line)}
  .div-big{font-size:38px;font-weight:700;line-height:1.1;color:var(--navy);margin-top:16px;letter-spacing:-.02em}
  .div-big small{font-size:13px;color:var(--muted);font-weight:500;margin-left:5px;letter-spacing:0}
  .div-detail{font-size:12.5px;line-height:1.6;color:var(--muted);margin-top:6px}
  .div-result{margin-top:auto;padding-top:14px;border-top:1px solid var(--line-soft);font-size:13px;line-height:1.55;color:var(--text)}
  .div-result b{color:var(--green)}
  .div-bottom{display:grid;grid-template-columns:1.25fr .75fr;gap:15px;flex:1;min-height:0}
  .div-note{
    display:flex;flex-direction:column;justify-content:center;padding:22px 24px;
    background:#FFFBF3;border:1px solid #F1E2C9;border-left:4px solid #C98A2E;border-radius:14px;
  }
  .div-note h4{font-size:15px;color:#8A5A13;font-weight:700;margin-bottom:8px}
  .div-note p{font-size:13px;line-height:1.75;color:#7A5A2A}
  .div-conclusion{
    display:flex;flex-direction:column;justify-content:center;padding:22px 24px;
    background:#F5F9FC;border:1px solid #D8E4F0;border-radius:14px;
  }
  .div-conclusion .status{align-self:flex-start;margin-bottom:10px}
  .div-conclusion p{font-size:13px;line-height:1.75;color:var(--text)}

  /* ===== Adaptation ===== */
  .adapt-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;flex:1;min-height:0}
  .adapt-card{border-radius:18px;border:1px solid var(--line);padding:32px 34px;display:flex;flex-direction:column;justify-content:center}
  .adapt-card.do{background:#F5FBF8;border-color:var(--green-line)}
  .adapt-card.dont{background:#F7F9FB}
  .adapt-card h3{font-size:20px;color:var(--navy);font-weight:700;margin-bottom:18px}
  .adapt-card ul{list-style:none;display:flex;flex-direction:column;gap:14px}
  .adapt-card li{position:relative;padding-left:24px;font-size:21px;line-height:1.5;color:var(--text);font-weight:600}
  .adapt-card li::before{content:"";position:absolute;left:0;top:10px;width:9px;height:9px;border-radius:50%}
  .adapt-card.do li::before{background:var(--green)}
  .adapt-card.dont li::before{background:#A9B4C0}
  .adapt-card .sub{font-size:12.5px;line-height:1.6;color:var(--muted);margin-top:8px}
  .verdict{
    flex:none;display:flex;align-items:center;justify-content:center;gap:24px;flex-wrap:wrap;
    background:var(--navy);color:#fff;border-radius:15px;padding:22px 28px;
    font-size:17px;font-weight:600;letter-spacing:.02em;
  }
  .verdict i{width:5px;height:5px;border-radius:50%;background:rgba(255,255,255,.4);flex:none}

  /* ===== Sample ===== */
  .sample-grid{display:grid;grid-template-columns:380px 1fr;gap:32px;align-items:center;flex:1;min-height:0}
  .sample-left{text-align:center;padding-right:32px;border-right:1px solid var(--line)}
  .sample-big{font-size:150px;line-height:.9;color:var(--navy);font-weight:700;letter-spacing:-.06em}
  .sample-left .unit{font-size:16px;color:var(--muted);margin-top:12px;font-weight:600}
  .dots{display:grid;grid-template-columns:repeat(10,1fr);gap:7px;max-width:280px;margin:24px auto 0}
  .dots i{aspect-ratio:1;border-radius:4px;background:var(--amber-bg);border:1px solid #EFD9B9}
  .can-cannot{display:flex;flex-direction:column;gap:14px}
  .cc-card{border-radius:16px;border:1px solid var(--line);padding:20px 24px}
  .cc-card.yes{background:#F5FBF8;border-color:var(--green-line)}
  .cc-card.no{background:#FFFBF3;border-color:#F1E2C9}
  .cc-card h3{display:flex;align-items:center;gap:10px;font-size:16px;color:var(--navy);font-weight:700;margin-bottom:10px}
  .cc-card h3 .status{font-size:10px;padding:4px 9px}
  .cc-card ul{list-style:none;display:flex;flex-direction:column;gap:7px}
  .cc-card li{position:relative;padding-left:18px;font-size:14px;line-height:1.55;color:var(--text)}
  .cc-card li::before{content:"";position:absolute;left:0;top:8px;width:6px;height:6px;border-radius:50%;background:#A9B4C0}
  .cc-card.yes li::before{background:var(--green)}
  .cc-card.no li::before{background:#C98A2E}

  /* ===== Illion ===== */
  .ref-panel{flex:1;min-height:0;display:flex;flex-direction:column;justify-content:center;border:1px solid #D8E4F0;border-radius:18px;background:linear-gradient(180deg,#FBFDFF 0%,#F4F9FC 100%);padding:30px 34px}
  .ref-head{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:22px}
  .ref-head .note{font-size:11px;color:var(--faint)}
  .flow{display:flex;align-items:center;justify-content:center;gap:16px;flex-wrap:nowrap;margin:8px 0 22px}
  .flow-box{flex:1;min-width:0;background:#fff;border:1px solid var(--line);border-radius:12px;padding:20px 16px;text-align:center;font-size:18px;font-weight:700;color:var(--navy)}
  .flow-eq{font-size:38px;line-height:1;color:var(--blue);font-weight:600;flex:none}
  .flow-arrow{font-size:28px;line-height:1;color:var(--faint);flex:none}
  .flow-result{flex:1.12;min-width:0;background:var(--blue-soft);border:1px solid #D4E3F3;border-radius:12px;padding:20px 16px;text-align:center;font-size:18px;font-weight:700;color:var(--blue)}
  .ref-panel > p{font-size:14.5px;line-height:1.75;color:var(--text);text-align:center;max-width:1000px;margin:0 auto}
  .ref-warn{display:flex;gap:10px;align-items:flex-start;margin-top:18px;padding:14px 16px;background:#FFFBF2;border-left:4px solid #D8A24A;border-radius:0 10px 10px 0;font-size:13px;line-height:1.65;color:#7A5118}
  .ref-warn b{color:#8A5A13;flex:none}
  .metric-chips{display:flex;gap:9px;justify-content:center;flex-wrap:wrap;margin-top:18px}
  .metric-chip{font-size:12px;padding:7px 13px;border-radius:999px;background:#fff;border:1px solid var(--line);color:var(--text);font-weight:600}

  /* ===== Steps ===== */
  .steps{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;flex:1;min-height:0}
  .step{display:flex;flex-direction:column;justify-content:center;background:#fff;border:1px solid var(--line);border-top:4px solid #D7E2EE;border-radius:16px;padding:26px 28px}
  .step .step-num{font-size:52px;line-height:1;color:#D7E2EE;font-weight:700;letter-spacing:-.03em;margin-bottom:18px}
  .step h3{font-size:21px;line-height:1.35;color:var(--navy);font-weight:700;margin-bottom:10px}
  .step p{font-size:14px;line-height:1.7;color:var(--text)}
  .chips{display:flex;gap:7px;flex-wrap:wrap;margin-top:20px}
  .chip{font-size:11px;padding:5px 10px;border-radius:999px;background:var(--soft);border:1px solid var(--line);color:var(--text);white-space:nowrap}

  /* ===== Summary table ===== */
  .table-card{flex:1;min-height:0;border:1px solid var(--line);border-radius:18px;overflow:hidden;background:#fff}

  /* ===== Model comparison (appendix) ===== */
  .cmp-cards{display:grid;grid-template-columns:1fr 1fr;gap:16px;flex:none}
  .cmp-card{
    display:flex;flex-direction:column;gap:13px;background:#fff;border:1px solid var(--line);
    border-top:4px solid var(--line);border-radius:16px;padding:17px 20px 15px;
  }
  .cmp-card.bscat{border-top-color:#2C6CB0;background:linear-gradient(180deg,#FBFDFF 0%,#fff 100%)}
  .cmp-card.illion{border-top-color:#A9680F;background:linear-gradient(180deg,#FFFDF9 0%,#fff 100%)}
  .cmp-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}
  .cmp-name{font-size:16px;line-height:1.3;color:var(--navy);font-weight:700}
  .cmp-sub{font-size:10.5px;line-height:1.4;color:var(--faint);margin-top:3px}
  .cmp-head .status{font-size:10px;padding:5px 9px}
  .cmp-stats{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;padding-top:11px;border-top:1px solid var(--line-soft)}
  .cmp-stat b{display:block;font-size:22px;line-height:1.15;color:var(--navy);font-weight:700;letter-spacing:-.01em;font-variant-numeric:tabular-nums}
  .cmp-stat span{display:block;font-size:9.5px;line-height:1.3;color:var(--faint);margin-top:3px;white-space:nowrap}
  .cmp-charts{display:grid;grid-template-columns:1.42fr 1fr;gap:16px;flex:1;min-height:0}
  .cmp-chart-card{
    display:flex;flex-direction:column;min-height:0;padding:14px 16px 12px;
    background:#FBFCFE;border:1px solid var(--line);border-radius:16px;
  }
  .cmp-chart-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex:none;margin-bottom:8px}
  .cmp-chart-title{font-size:13px;line-height:1.3;color:var(--navy);font-weight:700}
  .cmp-chart-note{font-size:9.5px;line-height:1.35;color:var(--faint);margin-top:3px}
  .cmp-legend{display:flex;align-items:center;justify-content:flex-end;gap:12px;flex-wrap:wrap;font-size:10px;color:var(--muted);white-space:nowrap}
  .cmp-legend .key{display:inline-flex;align-items:center;gap:5px}
  .cmp-legend i{width:9px;height:9px;border-radius:2px;display:inline-block;flex:none}
  .cmp-legend .bscat{background:#2C6CB0}
  .cmp-legend .illion{background:#A9680F}
  .cmp-plot{display:grid;grid-template-columns:28px 1fr;gap:7px;flex:1;min-height:0;padding-top:4px}
  .cmp-yaxis{display:flex;flex-direction:column;min-height:0}
  .cmp-yaxis-inner{position:relative;flex:1;margin-bottom:20px}
  .cmp-yaxis-inner span{position:absolute;right:0;transform:translateY(50%);font-size:9px;line-height:1;color:var(--faint);font-variant-numeric:tabular-nums}
  .cmp-plot-main{display:flex;flex-direction:column;min-height:0}
  .cmp-bars-area{position:relative;flex:1;min-height:0}
  .cmp-grid{position:absolute;left:0;right:0;height:1px;background:#E6EDF4;pointer-events:none}
  .cmp-groups{position:absolute;inset:0;display:grid;align-items:end;z-index:1}
  .cmp-group{height:100%;display:flex;align-items:flex-end;justify-content:center;min-width:0}
  .cmp-pair{display:flex;align-items:flex-end;justify-content:center;gap:2px;height:100%}
  .cmp-bar{position:relative;width:20px;height:0;flex:none;border-radius:4px 4px 0 0}
  .cmp-chart-card.bscat.single .cmp-bar{width:28px}
  .cmp-chart-card.illion.single .cmp-bar{width:24px}
  .cmp-bar.bscat{background:linear-gradient(180deg,#3D7FBF 0%,#2C6CB0 100%)}
  .cmp-bar.illion{background:linear-gradient(180deg,#BD7C22 0%,#A9680F 100%)}
  .slide.is-active .cmp-bar{animation:growH 1s cubic-bezier(.22,.61,.36,1) forwards}
  .cmp-val{
    position:absolute;left:50%;bottom:calc(100% + 4px);transform:translateX(-50%);
    font-size:9.5px;font-weight:700;line-height:1;color:var(--navy);white-space:nowrap;font-variant-numeric:tabular-nums;
  }
  .cmp-xaxis{height:20px;display:grid;align-items:center;font-size:9px;color:var(--faint);text-align:center;font-variant-numeric:tabular-nums}
  .cmp-callout{
    flex:none;margin-top:8px;padding:8px 10px;background:#fff;border:1px solid var(--line-soft);
    border-radius:9px;font-size:10.5px;line-height:1.5;color:var(--text);
  }
  .cmp-callout b{color:var(--navy)}
  .cmp-foot{
    flex:none;display:flex;align-items:flex-start;gap:11px;padding:12px 16px;
    background:#FFFBF3;border:1px solid #F1E2C9;border-left:4px solid #C98A2E;border-radius:12px;
  }
  .cmp-foot .mark{
    width:20px;height:20px;border-radius:50%;background:#F2E0C4;color:#8A5A13;
    font-size:12px;font-weight:800;display:grid;place-items:center;flex:none;margin-top:1px;
  }
  .cmp-foot b{display:block;font-size:11.5px;line-height:1.45;color:#8A5A13;letter-spacing:.02em}
  .cmp-foot p{font-size:10.5px;line-height:1.55;color:#7A5A2A;margin-top:2px}
  .cmp-source{display:block;margin-top:2px;font-size:9.5px;color:#9A7A4A}

  /* ===== Classification swap (appendix) ===== */
  .swap-kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;flex:none}
  .swap-kpi{
    display:flex;flex-direction:column;gap:5px;min-width:0;background:#fff;border:1px solid var(--line);
    border-top:4px solid var(--line);border-radius:14px;padding:12px 16px 10px;
  }
  .swap-kpi.auc{border-top-color:#2C6CB0}
  .swap-kpi.ks{border-top-color:#5B8FC4}
  .swap-kpi.gini{border-top-color:#7A8FA6}
  .swap-kpi.cap{border-top-color:#A9680F}
  .swap-kpi-label{font-size:9px;line-height:1;font-weight:700;letter-spacing:.1em;color:var(--faint)}
  .swap-kpi-pair{display:flex;align-items:flex-end;gap:7px;min-width:0}
  .swap-kpi-pair > div{display:flex;flex-direction:column;gap:1px;min-width:0}
  .swap-kpi-pair span{font-size:8.5px;line-height:1.1;color:var(--faint);white-space:nowrap}
  .swap-kpi-pair b,.swap-kpi-pair strong{
    font-size:22px;line-height:1.05;font-weight:700;letter-spacing:-.01em;color:var(--navy);
    font-variant-numeric:tabular-nums;
  }
  .swap-kpi-pair strong{color:var(--navy-2)}
  .swap-kpi-pair i{font-style:normal;font-size:13px;line-height:1;color:#A9B4C0;padding-bottom:3px}
  .swap-kpi-delta{font-size:10.5px;line-height:1;font-weight:700;color:#A8554C;font-variant-numeric:tabular-nums}
  .swap-table{display:flex;flex-direction:column;min-height:0;margin-top:2px}
  .swap-tr{
    display:grid;grid-template-columns:1.15fr .8fr .8fr .72fr;gap:6px;align-items:center;
    padding:8px 0;border-top:1px solid var(--line-soft);font-size:10px;line-height:1.2;
  }
  .swap-tr:first-child{border-top:0;padding:0 0 5px}
  .swap-th{font-size:8.5px;color:var(--faint);letter-spacing:.04em;font-weight:600}
  .swap-tr > span:nth-child(2),.swap-tr > span:nth-child(3),.swap-tr > span:nth-child(4){
    text-align:right;font-variant-numeric:tabular-nums;
  }
  .swap-tr b{font-size:10px;color:var(--navy);font-weight:700}
  .swap-tr .neg{color:#A8554C;font-weight:700}
  .swap-table-note{font-size:8.5px;line-height:1.35;color:var(--faint);margin-top:3px}
  .swap-insight{
    margin-top:9px;padding:8px 10px;background:#FFFBF3;border:1px solid #F1E2C9;
    border-left:3px solid #C98A2E;border-radius:8px;font-size:9.5px;line-height:1.45;color:#7A5A2A;
  }
  .swap-insight b{display:block;color:#8A5A13;font-size:9.5px;margin-bottom:2px}
  .swap-migrate{margin-top:auto;padding-top:10px;border-top:1px solid var(--line-soft)}
  .swap-migrate-head{display:flex;align-items:baseline;justify-content:space-between;gap:8px;margin-bottom:7px}
  .swap-migrate-title{font-size:9px;font-weight:700;letter-spacing:.08em;color:var(--faint);text-transform:uppercase}
  .swap-migrate-r{font-size:9px;color:var(--faint);font-variant-numeric:tabular-nums}
  .swap-migrate-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}
  .swap-migrate-grid > div{min-width:0;background:#fff;border:1px solid var(--line-soft);border-radius:8px;padding:7px 8px}
  .swap-migrate-grid b{display:block;font-size:15px;line-height:1.1;font-weight:700;color:var(--navy);font-variant-numeric:tabular-nums}
  .swap-migrate-grid span{display:block;font-size:8.5px;line-height:1.3;color:var(--faint);margin-top:3px}

  /* ===== Closing ===== */
  .closing{
    padding:66px 90px 46px;justify-content:space-between;
    background:
      radial-gradient(950px 560px at 82% 2%,#205074 0%,rgba(32,80,116,0) 62%),
      linear-gradient(135deg,#0B2B4A 0%,#0A2440 100%);
  }
  .closing .s-kicker{color:#9FC4E2}
  .closing h2{font-size:58px;line-height:1.15;letter-spacing:-.025em;color:#fff;font-weight:700;max-width:1150px;margin-top:14px}
  .closing .sub{font-size:18px;line-height:1.75;color:rgba(255,255,255,.64);margin-top:18px;max-width:950px}
  .takeaways{margin-top:34px;max-width:1120px}
  .takeaway{display:grid;grid-template-columns:130px 1fr;gap:22px;align-items:start;padding:17px 0;border-top:1px solid rgba(255,255,255,.14)}
  .takeaway .k{font-size:10.5px;letter-spacing:.14em;color:#8FB8DC;font-weight:700;text-transform:uppercase;padding-top:4px}
  .takeaway .v{font-size:19px;line-height:1.55;color:rgba(255,255,255,.9)}
  .takeaway .v b{color:#A8E7CE}
  .takeaway .v em{font-style:normal;color:#F3CE95}
  .closing-foot{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;padding-top:20px;border-top:1px solid rgba(255,255,255,.14)}
  .closing-foot .src{font-size:10.5px;line-height:1.6;color:rgba(255,255,255,.42)}
  .closing-foot .date{font-size:11px;color:rgba(255,255,255,.55);white-space:nowrap}

  /* ===== UI ===== */
  .deck-ui{position:fixed;right:24px;bottom:20px;display:flex;align-items:center;gap:7px;z-index:100}
  .deck-btn{
    width:41px;height:41px;border-radius:50%;border:1px solid rgba(11,43,74,.14);
    background:rgba(255,255,255,.9);color:var(--navy);cursor:pointer;
    display:grid;place-items:center;font-size:17px;line-height:1;
    -webkit-backdrop-filter:blur(12px);backdrop-filter:blur(12px);
    transition:background .2s,transform .2s,opacity .2s;
  }
  .deck-btn:hover{background:#fff;transform:translateY(-1px)}
  .deck-btn:focus-visible{outline:2px solid var(--blue);outline-offset:2px}
  .deck-btn:disabled{opacity:.32;cursor:default;transform:none}
  .counter{
    height:41px;display:flex;align-items:center;padding:0 15px;border-radius:999px;
    background:rgba(255,255,255,.9);border:1px solid rgba(11,43,74,.14);
    -webkit-backdrop-filter:blur(12px);backdrop-filter:blur(12px);
    font-size:11.5px;font-weight:700;letter-spacing:.08em;color:var(--navy);
  }
  .counter .cur{color:var(--navy)}
  .counter .sep{color:#B9C4CF;margin:0 5px}
  .counter .total{color:var(--faint)}
  .progress{position:fixed;left:0;right:0;bottom:0;height:3px;background:rgba(11,43,74,.08);z-index:100}
  .progress i{display:block;height:100%;width:0;background:linear-gradient(90deg,#2C6CB0,#5E8BB4);transition:width .35s ease}

  @media (prefers-reduced-motion:reduce){
    .slide{transition:none}
    .slide.is-active .reveal{animation:none;opacity:1;transform:none}
    .slide.is-active .fill,.slide.is-active .hc-fill,.slide.is-active .case-bar,.slide.is-active .cmp-bar{animation:none}
    .fill,.hc-fill{width:var(--w)}
    .case-bar{height:var(--h)}
    .cmp-bar{height:var(--h)}
  }

  @media (max-width:900px){
    .stage-wrap{padding:18px 18px 66px}
  }

  @media print{
    @page{size:1600px 900px;margin:0}
    html,body{overflow:visible;background:#fff;-webkit-print-color-adjust:exact;print-color-adjust:exact}
    .stage-wrap{position:static;display:block;overflow:visible;background:#fff;padding:0}
    .stage{transform:none!important;width:1600px;height:900px;border-radius:0;box-shadow:none;overflow:visible}
    .slide{position:relative;opacity:1!important;visibility:visible!important;transform:none!important;width:1600px;height:900px;page-break-after:always;break-after:page}
    .slide:last-child{page-break-after:auto;break-after:auto}
    .reveal{opacity:1!important;transform:none!important;animation:none!important}
    .fill,.hc-fill{width:var(--w)!important;animation:none!important}
    .case-bar{height:var(--h)!important;animation:none!important}
    .cmp-bar{height:var(--h)!important;animation:none!important}
    .deck-ui,.progress{display:none!important}
  }"""

NAV_JS = r"""(function () {
  var stage = document.getElementById('stage');
  var slides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  var total = slides.length;
  var current = -1;
  var prevBtn = document.getElementById('prevBtn');
  var nextBtn = document.getElementById('nextBtn');
  var curNum = document.getElementById('curNum');
  var totalNum = document.getElementById('totalNum');
  var progressFill = document.getElementById('progressFill');

  totalNum.textContent = String(total).padStart(2, '0');

  function fitStage() {
    var side = window.innerWidth < 900 ? 18 : 40;
    var bottom = window.innerWidth < 900 ? 66 : 76;
    var scale = Math.min(
      (window.innerWidth - side * 2) / 1600,
      (window.innerHeight - side - bottom) / 900
    );
    scale = Math.max(scale, 0.2);
    stage.style.transform = 'scale(' + scale + ')';
  }

  function show(index) {
    index = Math.max(0, Math.min(total - 1, index));
    if (index === current) return;
    slides.forEach(function (slide, i) {
      var active = i === index;
      slide.classList.toggle('is-active', active);
      slide.setAttribute('aria-hidden', active ? 'false' : 'true');
      if (active) {
        slide.removeAttribute('inert');
      } else {
        slide.setAttribute('inert', '');
      }
    });
    current = index;
    try {
      document.dispatchEvent(new CustomEvent('deck:slide', {detail: {index: index}}));
    } catch (e) {}
    curNum.textContent = String(current + 1).padStart(2, '0');
    progressFill.style.width = ((current + 1) / total * 100) + '%';
    prevBtn.disabled = current === 0;
    nextBtn.disabled = current === total - 1;
    try {
      history.replaceState(null, '', '#slide-' + (current + 1));
    } catch (e) {}
  }

  prevBtn.addEventListener('click', function () { show(current - 1); });
  nextBtn.addEventListener('click', function () { show(current + 1); });

  document.addEventListener('keydown', function (e) {
    var target = e.target;
    var onControl = target && target.closest && target.closest('button');
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === 'PageDown' ||
        (e.key === ' ' && !onControl)) {
      e.preventDefault();
      show(current + 1);
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp' || e.key === 'PageUp') {
      e.preventDefault();
      show(current - 1);
    } else if (e.key === 'Home') {
      e.preventDefault();
      show(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      show(total - 1);
    }
  });

  window.addEventListener('resize', fitStage);
  window.addEventListener('hashchange', function () {
    var m = window.location.hash.match(/slide-(\d+)/);
    if (m) show(parseInt(m[1], 10) - 1);
  });
  fitStage();

  var initial = 0;
  var match = window.location.hash.match(/slide-(\d+)/);
  if (match) {
    initial = Math.max(0, Math.min(total - 1, parseInt(match[1], 10) - 1));
  }
  show(initial);
})();"""
