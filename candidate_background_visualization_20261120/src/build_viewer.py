"""Build a local, offline four-panel frame viewer from frozen visualization data."""
from pathlib import Path
import hashlib
import json
import os
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
os.environ['XDG_CACHE_HOME'] = str(ROOT/'cache/xdg')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import cv2
import numpy as np


def main():
    frames = [json.loads(line) for line in (ROOT/'data/render_frames.jsonl').open()]
    compact = []
    for frame in frames:
        compact.append({'id':frame['pair_id'], 'index':frame['frame_index'],
            'time_s': frame['timestamp_s']-frames[0]['timestamp_s'],
            'add':{m:[frame['cameras'][c]['methods'][m]['add_mm'] for c in ('camera1','camera2')]
                   for m in ('R0','R1')}})
    delta = np.array([np.mean(f['add']['R1'])-np.mean(f['add']['R0']) for f in compact])
    selected = {'first':0, 'middle':len(frames)//2, 'last':len(frames)-1,
                'largest_pair_mean_improvement':int(np.argmin(delta)),
                'largest_pair_mean_worsening':int(np.argmax(delta))}
    for name, index in selected.items():
        pid = frames[index]['pair_id']
        images = [[cv2.imread(str(ROOT/f'overlays/{m}/{c}/{pid}.jpg')) for m in ('R0','R1')]
                  for c in ('camera1','camera2')]
        sheet = np.vstack([np.hstack(row) for row in images])
        target = ROOT/f'figures/preview_{name}_pair_{pid}.jpg'
        assert not target.exists()
        assert cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY,95])
    template = '''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>R0 OFF / R1 ON · 전체 프레임 키포인트와 포즈</title>
<style>
:root{color-scheme:light;font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif;color:#182c39;background:#edf2f5}
body{margin:0}main{max-width:1340px;margin:auto;padding:28px}h1{font-size:27px;margin-bottom:10px}h2{font-size:21px;margin:0 0 14px}p{line-height:1.7}
.card{background:white;border:1px solid #d9e2e9;border-radius:14px;padding:22px;margin:20px 0;box-shadow:0 4px 16px #15304c08}
.chips{display:flex;gap:10px;flex-wrap:wrap}.chip{background:#e5f2f2;border-radius:20px;padding:7px 13px;font-size:13px}.muted{color:#526878;font-size:14px}
a{color:#086b83}button,input,select{font:inherit}button{cursor:pointer;border:1px solid #b9ccd8;background:#f5fafc;border-radius:7px;padding:8px 12px}button:hover{background:#e5f2f2}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}input[type=range]{flex:1;min-width:220px;accent-color:#087e8b}input[type=number]{width:84px;padding:7px;border:1px solid #b9ccd8;border-radius:6px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{margin:0;background:#181c21;color:white;border-radius:9px;overflow:hidden}.panel img{display:block;width:100%;height:auto}.panel figcaption{padding:9px 12px;font-size:14px}
.column-title{font-weight:700;padding:8px 0}.chart{display:block;width:100%;height:auto}.stats{font-variant-numeric:tabular-nums;background:#f2f6f8;padding:12px;border-radius:8px;line-height:1.8}
table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:9px;border-bottom:1px solid #e3eaf0}code{font-size:13px;overflow-wrap:anywhere}
@media(max-width:700px){main{padding:12px}.card{padding:14px}.grid{grid-template-columns:1fr}.column-title{display:none}h1{font-size:22px}}
</style></head><body><main>
<h1>PnP 기반 초기화: 배경 OFF / ON 전체 프레임 비교</h1>
<p>UR_dual_2596의 통제 실험 <b>R0(배경 OFF)</b>와 <b>R1(배경 ON)</b>에서 저장된 최종 포즈를 표시합니다.
PnP와 단안 Candidates 정제로 만든 동일한 초기 포즈를 공유하며, 본 공동 최적화는 조건당 시작점 1개입니다.</p>
<div class="chips"><span class="chip">2,596쌍 · 카메라별 전체 프레임</span><span class="chip">10,384개 개별 이미지</span><span class="chip">R0/R1 모두 유효 포즈 5,192개</span><span class="chip">새 추론·최적화 없이 저장 결과 시각화</span></div>
<section class="card"><h2>키포인트와 로봇 베이스 좌표계</h2>
<p class="muted">주황 +: 원래 voting 입력점 · 노란 원: 최종 책임도로 선택한 MAP 후보 · 청록 점/선과 0–6 번호: 최종 포즈로 투영한 FK 골격.<br>
빨강 X / 초록 Y / 파랑 Z는 <b>추정된 로봇 베이스 좌표계</b>이며 축 길이는 모두 120 mm입니다. MAP는 사후 선택 후보이며, 혼합 우도를 단일 점 관측으로 바꿔 추정했다는 뜻은 아닙니다.</p>
<div class="controls"><button id="prev">← 이전</button><button id="next">다음 →</button><button id="play">자동 넘기기</button>
<label>프레임 <input id="number" type="number" min="1" max="2596" value="1"></label><input id="slider" type="range" min="1" max="2596" value="1" aria-label="전체 프레임 선택"></div>
<div class="controls"><button data-index="0">처음</button><button data-index="1298">중간</button><button data-index="2595">마지막</button>
<button data-index="__IMPROVE__">평균 ADD 개선 최대</button><button data-index="__WORSEN__">평균 ADD 악화 최대</button></div>
<div id="stats" class="stats"></div>
<div class="grid"><div class="column-title">R0 · 배경 OFF</div><div class="column-title">R1 · 배경 ON</div>
<figure class="panel"><a id="link_R0_camera1" href="overlays/R0/camera1/000000.jpg"><img id="R0_camera1" src="overlays/R0/camera1/000000.jpg" alt="R0 camera1 키포인트와 포즈"></a><figcaption>Camera 1 · R0</figcaption></figure>
<figure class="panel"><a id="link_R1_camera1" href="overlays/R1/camera1/000000.jpg"><img id="R1_camera1" src="overlays/R1/camera1/000000.jpg" alt="R1 camera1 키포인트와 포즈"></a><figcaption>Camera 1 · R1</figcaption></figure>
<figure class="panel"><a id="link_R0_camera2" href="overlays/R0/camera2/000000.jpg"><img id="R0_camera2" src="overlays/R0/camera2/000000.jpg" alt="R0 camera2 키포인트와 포즈"></a><figcaption>Camera 2 · R0</figcaption></figure>
<figure class="panel"><a id="link_R1_camera2" href="overlays/R1/camera2/000000.jpg"><img id="R1_camera2" src="overlays/R1/camera2/000000.jpg" alt="R1 camera2 키포인트와 포즈"></a><figcaption>Camera 2 · R1</figcaption></figure></div>
<p class="muted">이미지를 누르면 개별 원본 크기로 열립니다. 프레임 번호는 촬영 시간순 1–2,596이며 원래 pair ID는 000000–002595입니다. 두 카메라에는 각각의 촬영 시각에 해당하는 관절각/FK를 사용했습니다. 자동 넘기기는 검토용이며 실제 촬영 속도 재생이 아닙니다. 최대 개선/악화 버튼은 두 카메라 평균 ADD 차이로 선택한 시각화 사례로, 추정에 사용되지 않습니다.</p>
<noscript><p>프레임 이동에는 JavaScript가 필요합니다. 개별 이미지 폴더에서 모든 프레임을 직접 확인할 수도 있습니다.</p></noscript></section>
<section class="card"><h2>R0 · 배경 OFF: 전체 프레임 ADD</h2><a href="figures/R0_ADD_all_frames.png"><img class="chart" src="figures/R0_ADD_all_frames.png" loading="lazy" alt="R0 전체 프레임 ADD 에러 그래프"></a><p><a href="figures/R0_ADD_all_frames.svg">벡터 SVG</a> · <a href="overlays/R0/camera1/">Camera 1 이미지 폴더</a> · <a href="overlays/R0/camera2/">Camera 2 이미지 폴더</a></p></section>
<section class="card"><h2>R1 · 배경 ON: 전체 프레임 ADD</h2><a href="figures/R1_ADD_all_frames.png"><img class="chart" src="figures/R1_ADD_all_frames.png" loading="lazy" alt="R1 전체 프레임 ADD 에러 그래프"></a><p><a href="figures/R1_ADD_all_frames.svg">벡터 SVG</a> · <a href="overlays/R1/camera1/">Camera 1 이미지 폴더</a> · <a href="overlays/R1/camera2/">Camera 2 이미지 폴더</a></p></section>
<section class="card"><h2>같은 카메라에서 OFF / ON 비교</h2><a href="figures/R0_R1_ADD_comparison.png"><img class="chart" src="figures/R0_R1_ADD_comparison.png" loading="lazy" alt="카메라별 OFF ON ADD 비교"></a><p><a href="figures/R0_R1_ADD_comparison.svg">벡터 SVG</a> · <a href="data/add_by_frame.csv">전체 프레임 ADD CSV</a></p>
<p class="muted">가로축: 촬영 시간순 전체 프레임 번호. 세로축: 촬영 시각별 FK 7점 ADD(mm), 작을수록 좋습니다. 모든 그래프의 세로축 범위는 같고, 전 프레임을 생략·평활화·상단 절단 없이 표시했습니다. 점선은 50 mm 참고선입니다. 평가 기준은 기존 기록의 보정 변환과 FK이며 독립 외부 계측 GT는 아닙니다.</p></section>
<section class="card"><h2>저장 구조와 검증</h2><table><tr><th>항목</th><th>경로 / 내용</th></tr>
<tr><td>개별 이미지</td><td><code>overlays/R0 또는 R1/camera1 또는 camera2/PAIR_ID.jpg</code></td></tr>
<tr><td>정적 비교 예시</td><td><code>figures/preview_*.jpg</code> · 시작/중간/끝/최대 개선/최대 악화</td></tr>
<tr><td>생성 명령과 범례</td><td><a href="README.txt">README.txt</a></td></tr>
<tr><td>포즈·ADD 전수 검산</td><td><a href="validation/preparation.json">preparation.json</a></td></tr>
<tr><td>그림 파일·원본 보존</td><td><a href="validation/final_verification.json">final_verification.json</a></td></tr>
<tr><td>그래프 검사</td><td><a href="validation/add_graphs.json">add_graphs.json</a></td></tr></table>
<p class="muted">기존 모델·추정 코드·포즈·관측·평가 결과는 읽기 전용으로 사용했습니다. 새 파일은 candidate_background_visualization_20261120 안에만 생성했습니다. HTML은 같은 폴더의 이미지 파일을 사용하므로 폴더 구조를 유지해 주세요.</p></section>
<script>
const frames=__FRAMES__;
let index=0,timer=null;
const slider=document.getElementById('slider'),number=document.getElementById('number');
function show(value){index=Math.max(0,Math.min(frames.length-1,Number(value)||0));const f=frames[index];
slider.value=number.value=index+1;
for(const m of ['R0','R1'])for(const c of ['camera1','camera2']){const p=`overlays/${m}/${c}/${f.id}.jpg`;document.getElementById(`${m}_${c}`).src=p;document.getElementById(`link_${m}_${c}`).href=p;}
const mean=x=>(x[0]+x[1])/2;const delta=mean(f.add.R1)-mean(f.add.R0);
document.getElementById('stats').textContent=`프레임 ${index+1} / 2596 · pair ${f.id} · 첫 쌍 이후 ${f.time_s.toFixed(3)} s | Camera 1: OFF ${f.add.R0[0].toFixed(2)} → ON ${f.add.R1[0].toFixed(2)} mm | Camera 2: OFF ${f.add.R0[1].toFixed(2)} → ON ${f.add.R1[1].toFixed(2)} mm | 두 카메라 평균 차이(ON−OFF): ${delta>=0?'+':''}${delta.toFixed(2)} mm`;
}
slider.addEventListener('input',()=>show(Number(slider.value)-1));number.addEventListener('change',()=>show(Number(number.value)-1));
document.getElementById('prev').onclick=()=>show(index-1);document.getElementById('next').onclick=()=>show(index+1);
document.querySelectorAll('[data-index]').forEach(b=>b.onclick=()=>show(Number(b.dataset.index)));
document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='자동 넘기기';}else{timer=setInterval(()=>{if(index>=frames.length-1){clearInterval(timer);timer=null;document.getElementById('play').textContent='자동 넘기기';return;}show(index+1);},200);document.getElementById('play').textContent='정지';}};
document.addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowLeft')show(index-1);if(e.key==='ArrowRight')show(index+1);});
show(0);
</script></main></body></html>'''
    html = template.replace('__FRAMES__',json.dumps(compact,separators=(',',':'))).replace('__IMPROVE__',str(selected['largest_pair_mean_improvement'])).replace('__WORSEN__',str(selected['largest_pair_mean_worsening']))
    with (ROOT/'index.html').open('x') as f:f.write(html)
    with (ROOT/'data/preview_selection.json').open('x') as f:json.dump({k:{'frame_index':i+1,'pair_id':frames[i]['pair_id'],'pair_mean_R1_minus_R0_mm':float(delta[i])} for k,i in selected.items()},f,indent=2)
    print('Saved offline index.html and five 2x2 contact sheets.')


if __name__=='__main__':main()
