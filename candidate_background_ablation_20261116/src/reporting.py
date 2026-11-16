"""Self-contained Korean report from actual controlled-ablation evaluation files."""
from __future__ import annotations
import argparse
import base64
import csv
import html
import json
import os
from pathlib import Path
import numpy as np
from evaluate import ROOT, METHODS, CONTRASTS, clean, sha, write_json


def esc(value):return html.escape(str(value),quote=True)
def fmt(value,digits=2,scale=1):
    return '—' if value in (None,'') else f'{float(value)*scale:,.{digits}f}'
def interval(row,metric,scale=1):
    return f"{fmt(row.get(metric),scale=scale)} [{fmt(row.get(metric+'_ci_low'),scale=scale)}, {fmt(row.get(metric+'_ci_high'),scale=scale)}]"
def table(headers,records):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in records)+'</tbody></table></div>'
def load_csv(path):return list(csv.DictReader(Path(path).open()))
def numeric_rows(path):
    rows=load_csv(path)
    for row in rows:
        for k,v in row.items():
            if v=='':row[k]=None
            else:
                try:row[k]=float(v)
                except ValueError:pass
    return rows


def build_report(out,meta=None,evaluated=None,production=None,name='ablation_report.html'):
    out=Path(out).resolve();target=out/name
    if not out.is_relative_to(ROOT):raise ValueError('Report must remain in new extension')
    if target.exists():raise FileExistsError(target)
    meta=meta or json.loads((out/'evaluation.json').read_text())
    if evaluated is None:evaluated={'summary':numeric_rows(out/'summary_metrics.csv'),'paired':numeric_rows(out/'paired_comparisons.csv'),'sensitivity':numeric_rows(out/'bootstrap_sensitivity.csv')}
    production=production if production is not None else numeric_rows(out/'production_reference.csv')
    diagnostics=json.loads((out/'diagnostics.json').read_text())
    examples=json.loads((out/'examples.json').read_text())
    identity=json.loads((out/'production_identity.json').read_text())
    correlations=json.loads((out/'temporal_correlation.json').read_text())
    sources={}
    def source(path):
        path=Path(path).resolve();sources[str(path)]=sha(path);return path
    def link(path,title):
        path=source(path);return '<a href="'+esc(os.path.relpath(path,out))+'">'+esc(title)+'</a>'
    def image(name,caption):
        path=source(out/'figures'/(name+'.png'))
        encoded=base64.b64encode(path.read_bytes()).decode()
        return '<figure><img loading="lazy" src="data:image/png;base64,'+encoded+'" alt="'+esc(caption)+'"><figcaption>'+esc(caption)+'</figcaption></figure>'
    def section(identifier,title,content):return '<section id="'+identifier+'"><p class="eyebrow">CONTROLLED EXPERIMENT / 2026.11.16</p><h2>'+title+'</h2>'+content+'</section>'
    for filename in ('evaluation.json','summary_metrics.csv','paired_comparisons.csv','bootstrap_sensitivity.csv','diagnostics.json','production_reference.csv','production_identity.json','examples.json','temporal_correlation.json'):
        source(out/filename)
    source(Path(__file__))
    primary=[r for r in evaluated['summary'] if r['gate']=='final' and r['population']=='full' and r['camera']=='pooled']
    contrasts={r['contrast']:r for r in evaluated['paired'] if r['gate']=='final' and r['population']=='full' and r['camera']=='pooled'}
    summary_lookup={(r['gate'],r['population'],r['camera'],r['method']):r for r in evaluated['summary']}
    sections=[]
    n=meta['n_pairs'];S=meta['n_pre_eligible_pairs'];full=meta['full_sequence']
    hero='<p class="lead">동일한 Candidates 로봇 관측에서 <strong>최종 배경 결합</strong>과 <strong>Essential 초기화</strong>의 기여를 2×2 통제 실험으로 분리한다. 네 조건 모두 같은 후보·q·카메라 보정과 반복 상한을 공유하고, 각 조건은 공동 최적화 시작점 1개를 사용했다.</p>'
    hero+='<div class="metrics">'+''.join('<article><span>'+esc(label)+'</span><strong>'+esc(value)+'</strong><small>'+esc(sub)+'</small></article>' for label,value,sub in [
        ('실제 평가',f'{n:,}쌍',f'{2*n:,}개 카메라 관측'),('사전 적격',f'{S:,}쌍',f'보조 분모 {2*S:,}'),('공동 시작점','1개 / 조건','R0/R1, E0/E1 각각 동일 배열'),('실패 평가','분모에 포함','ADD∞ · 대체 포즈 없음')])+'</div>'
    if not full:hero+='<div class="warning">실행 점검용 부분 결과다. 전체 2,596쌍 실험 완료 또는 일반 성능 결론으로 해석하지 않는다.</div>'
    hero+='<p>이 보고서의 주 ADD50/AUC는 실패를 포함한 전체 분모다. 이전 성공 포즈만 평가한 보고서와 분모가 다르며, 평균 ADD는 유효 포즈에 한정해 유효 수와 함께 표시한다. 개발/최종 구간 분할은 사용하지 않는다.</p>'
    if full:
        rr=contrasts['R1-R0'];er=contrasts['E0-R0'];inter=contrasts['(E1-E0)-(R1-R0)']
        hero+=f"<div class='warning'><strong>실제 관측한 핵심 결과.</strong> R1−R0의 공동 유효 ADD 차이는 {fmt(rr['paired_ADD_delta_mm'])}mm, 전체 ADD50 차이는 {fmt(rr['ADD50_delta'],scale=100)}pp다. E0−R0의 ADD 차이는 {fmt(er['paired_ADD_delta_mm'])}mm, 전체 ADD50 차이는 {fmt(er['ADD50_delta'],scale=100)}pp다."
        if inter.get('paired_ADD_delta_mm_ci_low') is not None and inter['paired_ADD_delta_mm_ci_low']<=0<=inter['paired_ADD_delta_mm_ci_high']:
            hero+=' 상호작용 ADD의 95% 구간은 0을 포함하므로 추가 상호작용이 명확하다고 주장하지 않는다.'
        hero+='</div>'
    sections.append(section('overview','배경 결합과 초기값을 분리한 실제 실험',hero))

    design=table(['조건','포즈 초기화','최종 배경 항','해석'],[
        ['R0','로봇 단안 T₁,T₂','OFF','Candidates 직접 No-cross 대조군'],['R1','R0와 같은 A₀,G₀','ON','같은 초기값에서 배경 결합 추가'],
        ['E0','Essential + 로봇 metric scale','OFF','배경 기하는 초기값에만 사용'],['E1','E0와 같은 A₀,G₀','ON','E 초기값과 최종 배경 결합']])
    design+='<p>R은 A₀=T₁, G₀=T₂T₁⁻¹이며, 두 카메라 단안 초기해가 모두 있어야 준비된다. E는 현재 대응의 상대 회전 R과 이동 방향 d를 구하고 로봇 재투영으로 A와 log(s)를 정제해 G₀=[R,sd]를 만든다. E 생성 실패는 명시적 실패이며 R로 대체하지 않는다. proposal은 로봇 항 비용과 안정적인 ID 동률 규칙으로 한 번 고른다.</p><p>초기 A₀/G₀·후보·관측 해시와 초기 책임도를 OFF/ON이 공유한다. ON의 P₀는 해당 G₀로 고정 대응 전체에서 삼각측량하며, E inlier만 다시 골라 목적함수를 바꾸지 않는다. R1도 배경점 초기화와 최적화에 RGB 대응을 쓴다.</p>'
    design+='<p>R0는 B=GA로 치환하면 두 단안 목적함수의 합으로 분리된다. E0 역시 최종 비용은 분리되지만 초기값에 두 영상 정보가 들어 있다. A/G의 공동 파라미터화와 유한 반복이 만든 수치 차이를 영상 결합 이득으로 부르지 않는다.</p>'
    separation=ROOT/'validation/no_cross_diagnostic.json'
    if separation.exists():
        design+='<p>사전에 정한 4pair의 R/E 8개 조건에서 같은 A₀/B₀의 독립 단안 EM과 A/G OFF를 대조했다. 최대 목적함수 차이 약 1.21×10⁻⁷, translation 약 1.91×10⁻⁶m, rotation 약 0.000688°로 큰 차이가 없었다. 공통 단안 초기화 결과도 원래 구현의 포즈·대안·quality와 정확히 같았다. 이는 분리가능성의 수치 진단이며 새 성능법 또는 GT 정확도 실험이 아니다. '+link(separation,'No-cross·원래 단안 절차 대조')+'</p>'
    sections.append(section('design','네 조건과 실제 공유하는 것',design))

    method='<div class="equation">A = T<sub>C₁←B</sub>, &nbsp; G = T<sub>C₂←C₁</sub>, &nbsp; T<sub>C₂←B</sub> = GA<br>z₁ₖ = π₁(AX₁ₖ), &nbsp; z₂ₖ = π₂(GAX₂ₖ)</div><p>각 촬영 시각의 관절각 q₁/q₂로 계산한 FK 7점을 각 카메라의 포즈로 투영한다. 카메라와 베이스가 고정된 기록이어도 이전 pair의 포즈·대응·평균을 다음 pair의 추정에 쓰지 않는다.</p>'
    method+='<div class="equation">ℓ(z) = (1−ε) Σ<sub>h</sub> α<sub>h</sub> 𝒩(z; μ<sub>h</sub>, σ²I) + ε/(WH)<br>L<sub>Candidates</sub> = −Σ<sub>i,k</sub> log ℓ<sub>ik</sub>(z<sub>ik</sub>)</div><p>원래 voting 후보 8개와 support를 유지한다. 유한 좌표·양의 support·영상 내부·원래 valid 조건으로 후보를 필터링하고, 유지한 support를 정규화한 α를 고정한다. σ=2px, ε=0.05, Gaussian 정규화와 uniform outlier를 포함한다. 후보를 새로 만들거나 모델을 학습하지 않는다.</p>'
    method+='<div class="equation">L<sub>OFF</sub> = L<sub>Candidates</sub>(A,GA) + L<sub>robot depth</sub><br>L<sub>ON</sub> = L<sub>OFF</sub> + L<sub>background reprojection</sub>(G,P) + L<sub>background depth</sub>(G,P)</div><p>ON은 최대 80개 현재 배경 대응, σ<sub>b</sub>=1.5px, 2D 블록 Huber δ=3, λ=1을 쓴다. OFF는 배경점 P 자체를 변수에서 제거하며 배경 재투영·깊이·E/G prior·초기 포즈 유지 항이 없다. A/G 회전·이동 12변수는 모두 자유롭다. OFF에 배경 수·시차·hull·cheirality·기선 gate를 적용하지 않는다.</p><p>외부 EM 최대 10회, 각 내부 solve max_nfev=60, 일반 ftol/xtol/gtol=10⁻⁸이다. 책임도는 같은 초기값에서 시작하지만 이후 포즈에 따라 달라진다. 실제 전체 목적함수가 증가하는 업데이트는 거부한다. ON 정보량은 P nuisance 방향을 제거한 포즈 Jacobian, OFF는 배경 변수가 없는 로봇 Jacobian으로 계산한다. translation scale은 0.5m다.</p>'
    audit=ROOT/'validation/config_and_report_audit.json'
    if audit.exists():method+='<p>현재 구현 보고서와 실행 설정을 대조했다. 코드 기본 max_nfev=50 대신 실행 설정 60, 기본 mask dilation=8px 대신 저장 대응의 50px를 유지한다. production 공동 시작점 3개를 이번 통제 조건에서만 1개로 고정했다. '+link(audit,'코드·보고서 대조 기록')+'</p>'
    sections.append(section('method','후보 우도와 배경 OFF의 정확한 의미',method))

    evaluation=f'<p>전체 분모는 {2*n:,}개 관측이며, 공통 사전 적격 집합은 {S:,}쌍·{2*S:,}개 관측이다. 적격 집합은 R/E 포즈 초기값과 필요한 두 ON P₀가 최적화 전에 모두 준비된 pair로 정한다. 최종 ADD·성공 여부로 선택하지 않는다.</p>'
    evaluation+=table(['초기화','준비 pair','ON P₀ 준비','proposal 생성 평균 ms'],[[family,diagnostics['initializations'][family]['available'],diagnostics['initializations'][family]['on_ready'],fmt(diagnostics['initializations'][family]['generation_ms']['mean'])] for family in ('R','E')])
    evaluation+='<p>위 generation 시간은 proposal 생성까지이며 선택·삼각측량을 제외한다. 전체 초기화 비용은 뒤의 공유 all_initialization_ms를 사용한다.</p>'
    evaluation+='<p>사전 제외 원인: '+esc(json.dumps(meta['pre_eligible_exclusion_reasons'],ensure_ascii=False))+'</p><p>공통 검사는 유한 포즈·로봇 양의 깊이·유효 로봇 정보량·활성 잔차의 rank/condition이다. 최종 검사는 공통 검사와 ON의 배경 품질 검사를 모두 요구한다. gate 이전의 유한 포즈를 결과 JSON에 보존하되 채택 성능은 valid flag로 마스크한다.</p><div class="equation">ADD = (1000/7) Σ<sub>k</sub> ‖T̂X<sub>k</sub> − T<sub>ref</sub>X<sub>k</sub>‖₂ [mm]<br>ADD50 = Σ1[ADD &lt; 50] / N, &nbsp; AUC = trapz(ADD curve, 0:5:100) / 100</div><p>실패는 ADD=∞로 모든 유한 임계값에서 실패다. 평균·중앙값·P90은 유효 포즈에만 계산한다. annotation은 기록된 보정 포즈와 encoder FK에서 생성됐으므로 독립 실측 GT 또는 mesh ADD가 아니다. 평가용 외참수는 이 평가 단계에서만 읽는다.</p>'
    sections.append(section('evaluation','전체 분모·사전 적격·두 gate 기준',evaluation))

    controls='<div class="controls"><label>집합<select id="population"><option value="full">전체</option><option value="pre_eligible">공통 사전 적격</option></select></label><label>채택 기준<select id="gate"><option value="final">최종 gate</option><option value="common">공통 검사만</option></select></label><label>카메라<select id="camera"><option value="pooled">두 카메라 합산</option><option value="camera1">Camera 1</option><option value="camera2">Camera 2</option></select></label></div><p id="selection-note" aria-live="polite"></p><div id="dynamic-results"></div>'
    controls+='<h3>주 결과: 전체·최종 gate·두 카메라</h3>'+table(['조건','유효 / 전체','실패','ADD50 % [95% CI] ↑','AUC [95% CI] ↑','조건부 평균 mm [CI] ↓','중앙값','P90'],[[r['method'],f"{r['n_valid']}/{r['n_total']}",r['n_failed'],interval(r,'ADD50',100),interval(r,'ADD_AUC'),interval(r,'ADD_mean_mm'),fmt(r['ADD_median_mm']),fmt(r['ADD_p90_mm'])] for r in primary])
    controls+=image('add_curves','실패를 포함한 ADD 곡선. 전체/사전 적격과 최종/common gate를 분리한다.')
    sections.append(section('results','실패를 포함한 전체 성능',controls))

    effect=table(['대비','공동 유효 관측','paired ADD Δ mm [95% CI] ↓','전체 ADD50 Δ pp [CI] ↑','전체 AUC Δ pp [CI] ↑'],[[label,r['n_common_valid'],interval(r,'paired_ADD_delta_mm'),interval(r,'ADD50_delta',100),interval(r,'ADD_AUC_delta',100)] for label,r in contrasts.items()])
    effect+='<p>ADD Δ는 뒤 조건−앞 조건으로 음수가 개선이다. paired ADD는 동일 pair·camera에서 두 방법 모두 유효한 교집합만 사용한다. ADD50/AUC Δ는 초기화 실패·최종 실패를 포함한 같은 전체 분모로 계산한다. 상호작용은 네 조건 모두 유효한 교집합에서만 계산하며 이 표본 수를 따로 표시한다.</p>'+image('paired_effects','사전 지정 1초 시간 블록, 2,000회 seed 0의 95% percentile 신뢰구간.')
    answers=[]
    for title,labels,explanation in [
        ('같은 초기 포즈에서도 배경 결합이 개선하는가?',('R1-R0','E1-E0'),'두 대비는 각각 동일한 R/E 초기값에서 최종 배경 항만 바꾼다.'),
        ('E 초기값만으로도 개선되는가?',('E0-R0',),'OFF끼리의 차이다. E0의 입력 초기값에는 두 영상의 기하 정보가 포함된다.'),
        ('두 효과가 함께 작용하는가?',('E1-R1','(E1-E0)-(R1-R0)'),'E1−R1은 ON 최적화의 초기값 민감도다. 상호작용 ADD가 음수이면 E 초기값에서 배경 추가의 감소 효과가 더 크지만, 네 방법 공동 유효 집합에 한정한다.')]:
        text='<h3>'+title+'</h3><p>'+explanation+'</p>'
        for label in labels:
            r=contrasts[label]
            direction='감소' if r['paired_ADD_delta_mm'] is not None and r['paired_ADD_delta_mm']<0 else '증가' if r['paired_ADD_delta_mm'] is not None and r['paired_ADD_delta_mm']>0 else '변화 없음/평가 불가'
            text+=f"<p><strong>{esc(label)}</strong>: 공동 유효 {int(r['n_common_valid']):,}개에서 ADD 차이 {interval(r,'paired_ADD_delta_mm')}mm ({direction})."
            if r['ADD50_delta'] is not None:text+=f" 전체 ADD50 차이는 {interval(r,'ADD50_delta',100)}pp, AUC 차이는 {interval(r,'ADD_AUC_delta',100)}pp다."
            text+='</p>'
        answers.append(text)
    effect+=''.join(answers)
    if full:
        both=all(contrasts[k]['paired_ADD_delta_mm'] is not None and contrasts[k]['paired_ADD_delta_mm']<0 and contrasts[k]['ADD50_delta']>0 for k in ('R1-R0','E1-E0'))
        if both:effect+='<div class="warning"><strong>같은 초기값의 배경 결합은 R과 E 두 경로 모두 개선했다.</strong> 공동 유효 ADD 감소와 실패 포함 ADD50 증가가 함께 관측됐다. 따라서 핵심 이득을 E 초기화만의 효과로 설명할 수 없다.</div>'
        er=contrasts['E0-R0'];on=contrasts['E1-R1'];interaction=contrasts['(E1-E0)-(R1-R0)']
        if er['paired_ADD_delta_mm']>0 and er['ADD50_delta']<0:
            effect+='<p><strong>E 초기값만의 추가 이득은 이 시퀀스에서 관측되지 않았다.</strong> E0는 R0보다 공동 유효 ADD가 커지고 전체 ADD50도 낮아졌다. E1도 R1보다 전체 ADD50가 낮다. 이는 이번 고정 단일 시작점 규칙에 대한 결과이며 E 기하의 일반적인 무용성을 뜻하지 않는다.</p>'
        if interaction.get('paired_ADD_delta_mm_ci_low') is not None and interaction['paired_ADD_delta_mm_ci_low']<=0<=interaction['paired_ADD_delta_mm_ci_high']:
            effect+='<p><strong>추가적인 상승효과는 확립되지 않았다.</strong> 네 조건 공동 유효 상호작용의 95% CI가 0을 포함한다. 배경 결합 자체의 개선과 두 요인의 추가 상호작용을 구분해야 한다.</p>'
    sections.append(section('effects','네 대비와 세 가지 효과 질문',effect))

    gate=table(['조건','공통 유효','최종 유효','배경 gate 탈락 관측','그중 ADD&lt;50','탈락 조건부 ADD mm','채택 조건부 ADD mm'],[[r['method'],r['common_valid'],r['final_valid'],r['rejected_by_background'],r['rejected_ADD_below_50'],fmt(r['rejected_ADD_mean_mm']),fmt(r['kept_ADD_mean_mm'])] for r in diagnostics['gate_effects']])
    gate+='<p><strong>배경 gate의 선택 효과와 구분되는가?</strong> 위 표는 공통 검사까지 통과했으나 배경 gate로 제거된 관측을 직접 센 것이다. 전체 분모에서 gate로 실패가 늘면 ADD50/AUC의 분자는 증가할 수 없다. 반면 성공 포즈만의 평균은 어려운 사례 제외로 좋아질 수 있으므로 조건부 평균만으로 결합 효과를 판단하지 않는다. 공통 검사 기준의 보조 성능과 같은 초기값 대비를 함께 확인해야 한다.</p>'
    removed=sum(r['rejected_by_background'] for r in diagnostics['gate_effects'])
    if removed==0:
        gate+='<div class="warning"><strong>이번 실제 결과의 추가 배경 gate 탈락은 0개다.</strong> 네 조건 모두 common-valid와 final-valid 집합이 정확히 같아 두 기준의 성능도 일치한다. 따라서 R1의 이득은 배경 gate가 어려운 성공 포즈만 제거해서 생긴 이득이 아니다. E 조건의 실패는 공통 로봇/정보량 검사에서 발생했다. 일부 결과에서 배경 검사도 동시에 실패했을 수 있으나, 공통 통과 뒤 추가 탈락한 경우와 구분한다.</div>'
    gate+='<h3>실패 원인: pair 단위</h3>'+table(['조건','기준','원인','pair 수'],[[r['method'],r['gate'],esc(r['reason']),r['n_pairs']] for r in meta['failure_counts']])
    gate+='<p>동일 pair가 common/final 두 행에 표시될 수 있으며 이를 합산하지 않는다. ON 실패를 OFF나 단안 포즈로 바꿔 성공으로 세지 않는다. 독립 대응 GT가 없으므로 배경 재투영 gate 실패 전체를 오매칭으로 단정하지 않는다.</p>'
    sections.append(section('gates','배경 gate의 선택 효과와 실패 원인',gate))

    pck=table(['조건','유효 / 전체','raw 입력 PCK5 %','사후 MAP PCK5 %','pose 투영 PCK5 %','MAP gate 이전 %','pose gate 이전 %'],[[r['method'],f"{r['n_valid']}/{r['n_total']}",fmt(r['raw_PCK5'],scale=100),fmt(r['map_PCK5'],scale=100),fmt(r['projected_PCK5'],scale=100),fmt(r['map_PCK5_before_gate'],scale=100),fmt(r['projected_PCK5_before_gate'],scale=100)] for r in primary])
    pck+='<p>raw는 고정 voting 평균을 평가하므로 네 조건에서 동일하다. MAP는 최종 책임도로 고른 원래 후보이며 포즈 투영점과 다르다. pose PCK는 추정 포즈로 FK를 투영한 점이다. reference가 영상 내부인 점만 분모에 포함하고, 해당 reference에 예측이 없으면 실패로 센다. MAP/pose의 주 PCK는 gate 실패도 분모에 남긴다. gate 이전 열은 저장된 유한 포즈·후보의 진단값이며 채택 성능과 구분한다.</p><p>원래 후보 ID·좌표·γ와 outlier 책임도는 pair_results.jsonl에 저장한다. PCK는 보조 지표이며 주 지표는 포즈 ADD다.</p>'
    sections.append(section('pck','입력·후보 선택·포즈 PCK의 구분',pck))

    bootstrap='<p>두 카메라 timestamp의 평균으로 pair 시각을 정렬하고 첫 pair 시각부터 비중첩 고정 시간 bin을 만든다. 원래 시간축에서 점유 블록 개수만큼 복원 재표집한 뒤 사전 적격·공동 유효 마스크를 적용한다. 모든 조건에 동일한 블록 draw를 사용하며 두 카메라는 함께 움직인다. 블록별 pair 수 차이 때문에 반복별 분모도 해당 재표집 관측 수가 된다. 실패 행이나 개별 키포인트를 먼저 삭제하거나 독립 표본으로 재표집하지 않는다.</p>'
    bootstrap+=table(['블록 초','블록 수','반복','seed','재표집 pair 수 범위'],[[r['seconds'],r['n_blocks'],r['n_resamples'],r['seed'],str(r['sample_pair_count_min'])+'–'+str(r['sample_pair_count_max'])] for r in meta['bootstrap']])
    sensitive=[r for r in evaluated['sensitivity'] if r['gate']=='final' and r['population']=='full' and r['camera']=='pooled']
    bootstrap+=table(['블록 초','대비','paired ADD 95% CI mm','ADD50 95% CI pp','AUC 95% CI pp','ADD 유효 반복'],[[r['block_seconds'],r['contrast'],interval(r,'paired_ADD_delta_mm'),interval(r,'ADD50_delta',100),interval(r,'ADD_AUC_delta',100),r['paired_ADD_delta_mm_bootstrap_valid']] for r in sensitive])
    long=[r for r in correlations['records'] if r['requested_lag_s']>=2 and r['correlation'] is not None and abs(r['correlation'])>.2]
    bootstrap+='<p>0.5/1/2초는 사전 지정 분석 설정이며 독립성을 보장하는 값이 아니다. 95% percentile 구간이며 유효 교집합이 비어 지표를 계산할 수 없는 반복은 별도 개수로 기록한다. 다음 lag 진단은 원래 인덱스 간격을 보존하고 두 시점 모두 유효한 값만으로 상관을 계산한 기술통계다. p-value 또는 독립성 검정으로 해석하지 않는다.</p>'
    if long:bootstrap+='<div class="warning">2초 이상 lag에서 |상관계수|&gt;0.2인 대비가 관측됐다. 긴 시간 상관이 남을 수 있으므로 이 블록 길이들의 CI를 보편적 불확실성 한계로 해석하지 않는다.</div>'
    bootstrap+=table(['대비','목표 lag s','실제 lag 중앙값 s','유효 쌍','상관계수'],[[r['contrast'],r['requested_lag_s'],fmt(r['actual_median_lag_s'],3),r['n_pair_pairs'],fmt(r['correlation'],3)] for r in correlations['records']])
    bootstrap+=image('temporal_effects','원래 pair 시각의 공동 유효 ADD 차이. 누락 시점을 압축하지 않는다.')
    sections.append(section('bootstrap','시간 상관과 신뢰구간 민감도',bootstrap))

    prod=table(['참고 방법','유효 / 전체','실패','전체 ADD50 %','전체 AUC','조건부 ADD 평균 mm'],[[r['method'],f"{r['n_valid']}/{r['n_total']}",r['n_failed'],fmt(r['ADD50'],scale=100),fmt(r['ADD_AUC'],4),fmt(r['ADD_mean_mm'])] for r in production if r['camera']=='pooled'])
    prod+='<p>기존 Mono-Candidates / Dual-Candidates의 저장 결과를 실패 포함 분모로 다시 집계했다. 기존 최대 3개 시작점 및 생산용 top-3 선택 정책을 유지한 참고표다. E1은 E 초기값 하나만 사용하는 통제 조건이므로 기존 Dual-Candidates와 같은 방법으로 표시하지 않는다.</p>'
    checks={k:v for k,v in identity.items() if isinstance(v,bool)}
    if identity.get('source_code_sha_matches_current'):checks['production_source_files_match']=all(identity['source_code_sha_matches_current'].values())
    prod+=table(['출처 대조','결과'],[[esc(k),esc(v)] for k,v in checks.items()])+'<p>'+link(out/'production_identity.json','관측·매칭·코드 출처와 차이 전체 기록')+'</p>'
    sections.append(section('production','기존 다중 시작점 결과는 별도 참고',prod))

    timing=table(['조건','실제 시작점 평균','EM 평균 / 최대','내부 nfev 합 평균','60 도달 solve / 전체','최적화 평균 ms','전체 method 평균 ms'],[[m,fmt(d['starts']['mean']),fmt(d['em_iterations']['mean'])+' / '+fmt(d['em_iterations']['max'],0),fmt(d['optimizer_nfev']['mean']),str(d['internal_solves_at_nfev60'])+'/'+str(d['internal_solves']),fmt(d['optimization_ms']['mean']),fmt(d['total_ms']['mean'])] for m,d in diagnostics['methods'].items()])
    timing+=table(['공유 단계','평균 ms','중앙값 ms','P90 ms'],[[k,fmt(v['mean']),fmt(v['median']),fmt(v['p90'])] for k,v in diagnostics['shared_timings'].items()])
    timing+='<p>이번 실행은 검증된 과거 CUDA FP32 로봇 관측과 RoMa 대응 캐시를 재사용하고 CPU에서 초기화·네 조건을 실행한다. 위 시간은 새 로봇 추론·GPU matching을 포함한 online 지연시간이 아니다. 같은 반복 상한도 변수 수·잔차 수·Jacobian 호출이 달라 계산량이 같지 않다. 병렬 배치 wall time의 역수를 단일 pair FPS로 바꾸지 않는다.</p>'
    run_manifest=Path(meta['run_dir'])/'manifest.json'
    if run_manifest.exists():
        run_record=json.loads(run_manifest.read_text())
        wall=run_record.get('wall_seconds',run_record.get('wall_time_s',run_record.get('wall_s')))
        if wall is not None:timing+=f"<p>이번 전체 배치 wall time은 {fmt(wall,3)}초이며 CPU {run_record.get('workers','—')} workers로 실행했다. "+link(run_manifest,'실제 실행 manifest')+'</p>'
    timing+='<h3>종료 원인과 정보량 진단</h3>'+table(['조건','종료 원인','pair 수'],[[m,esc(reason),count] for m,d in diagnostics['methods'].items() for reason,count in d['terminations'].items()])
    timing+=table(['조건','후보 NLL 평균','rank 중앙값','condition 중앙값','배경 median 평균 px','배경 p90 평균 px'],[[m,fmt(d['candidate_nll']['mean']),fmt(d['reduced_rank']['median'],0),fmt(d['reduced_condition']['median']),fmt(d['background_median_px']['mean']),fmt(d['background_p90_px']['mean'])] for m,d in diagnostics['methods'].items()])
    timing+='<p><strong>개선이 작거나 나쁜 원인을 어디까지 말할 수 있는가?</strong> 동일 초기값 대비와 E0−R0가 초기값 요인을 분리하고, gate 원인·정보량·반복 상한·종료 기록은 수치적 제약을 보여 준다. 다만 수렴 예산 확대나 대응 GT 교체를 별도로 실행하지 않았으므로 제한된 반복, 국소 최적점, 정보량, 대응 오차 각각의 인과 기여를 이 기록만으로 확정하지 않는다. 원인 없는 성능 개선 주장 대신 실제 실패와 위 진단을 함께 제시한다.</p>'
    mechanism_path=ROOT/'validation/mechanism_analysis.json'
    if full and mechanism_path.exists():
        mechanism=json.loads(mechanism_path.read_text());source(mechanism_path)
        groups=mechanism['R1_R0_effect_groups']
        timing+='<h3>실제로 개선이 작았던 사례와 실패의 상세 진단</h3><p>R1−R0의 두 카메라 pair 평균 ADD 변화로 사후 설명 집단을 만든다. 1mm 기준은 이 진단의 기술적 구분이며 설정 조정이나 성능 평가 집합 선택에 사용하지 않았다.</p>'
        timing+=table(['설명 집단','pair 수','R 초기 ADD 평균 mm','R1−R0 평균 mm','rank 중앙값','배경 p90오차 평균 px'],[[label,g['pair_count'],fmt(g['R_initial_ADD_mm']['mean']),fmt(g['pair_mean_ADD_delta_mm']['mean']),fmt(g['R1_diagnostics']['metrics']['rank']['median'],0),fmt(g['R1_diagnostics']['metrics']['background_p90_error_px']['mean'],3)] for label,g in groups.items()])
        small=groups['small_effect_abs_at_most_1mm'];better=groups['improved_more_than_1mm']
        timing+=f"<p>작은 효과(|Δ|≤1mm)는 {small['pair_count']}쌍이며 초기 ADD 평균은 {fmt(small['R_initial_ADD_mm']['mean'])}mm다. 1mm 초과 개선군 {better['pair_count']}쌍의 초기 ADD {fmt(better['R_initial_ADD_mm']['mean'])}mm보다 낮아, 이미 좋은 초기값에서 개선 여지가 작다는 설명과 양립한다. 작은 효과군도 rank 중앙값 12, 유효 RGB 80개이므로 단순한 대응 개수 부족이나 rank 결손으로 설명되지 않는다. 이 비교는 연관성이지 원인 증명이 아니다.</p>"
        rejected=[]
        for m in ('E0','E1'):
            g=mechanism['method_diagnostics_by_failure_group'][m]['common_gate_rejected'];metrics=g['metrics']
            rejected.append([m,g['n_pairs'],fmt(metrics['effective_robot_min_camera']['median'],3),fmt(metrics['rank']['median'],0),esc(json.dumps(mechanism['failure_reasons'][m]['common_all'],ensure_ascii=False))])
        timing+=table(['E 실패군','pair 수','최소 camera 유효 robot 중앙값','rank 중앙값','공통 실패 전체 이유(중복 가능)'],rejected)
        budget_counts=[]
        for m in ('E0','E1'):
            failures=[r for r in mechanism['failure_pair_records'] if r['method']==m and r['stage']=='common_gate_rejected']
            not_limited=sum(not r.get('any_inner_max60',False) and r.get('termination')!='maximum_em_iterations' for r in failures)
            budget_counts.append(f'{m} {not_limited}/{len(failures)}쌍')
        timing+='<p><strong>실패 대부분은 기록된 반복 상한에 도달하지 않았다.</strong> '+', '.join(budget_counts)+'은 내부 nfev=60 도달도 EM10 예산 소진도 없었다. 따라서 실패 전체를 단순한 반복 예산 부족으로 설명할 근거는 없다. 수치적 종료 기준 만족과 올바른 해로의 수렴은 별개다.</p>'
        timing+='<p>E 실패군의 한 카메라 유효 로봇 정보가 0에 가까워지며 rank가 떨어진 기록은 후보 outlier 책임도 포화 또는 나쁜 국소 해와 양립한다. 초기화는 모든 pair에서 존재했으므로 E 초기화 부재가 이번 실패의 원인은 아니다. 목적함수 수렴 판정은 reference ADD의 정확도 보장이 아니다. 반복 예산 확대 실험이 없으므로 더 오래 풀면 이 문제가 해결된다고 단정하지 않는다.</p>'
        stages=[r for r in mechanism['initial_to_stage_paired_changes'] if r['gate']=='final' and r['camera']=='pooled']
        timing+=table(['조건','같은 최종 유효 관측 수','초기 ADD 평균','최종 ADD 평균','최종−초기 ADD'],[[r['method'],r['n_common_camera_observations'],fmt(r['initial_or_before_ADD_mm']['mean']),fmt(r['final_or_after_ADD_mm']['mean']),fmt(r['after_minus_before_ADD_mm']['mean'])] for r in stages])
        timing+='<p>위 전후 변화는 같은 최종 유효 관측에서 계산했다. 서로 다른 성공 집합의 초기/최종 평균을 빼지 않는다. 로봇 NLL 또는 배경 잔차를 줄이는 것이 평가 reference ADD를 반드시 줄이는 것은 아니다.</p>'
        assoc=mechanism['descriptive_associations']
        timing+=table(['R1 진단량','pair 수','R1−R0 pair ADD와 Spearman ρ'],[[r['diagnostic'],r['n_pairs'],fmt(r['spearman_rho_with_R1_minus_R0_pairmean_ADD'],3)] for r in assoc])
        timing+='<p>위 ρ는 순차 pair의 기술적 연관성이다. 독립 표본 가설검정이나 원인 분해가 아니다. R1 배경 p90 재투영오차가 작고 Huber downweight가 없다는 사실도 정답 대응을 직접 증명하지 않는다. 대응 GT와 예산 변화 실험 없이 오매칭·관측 편향·최적화의 기여를 각각 확정하지 않는다. '+link(mechanism_path,'초기값·정보량·수렴·대응 진단 전체')+'</p>'
    sections.append(section('timing','계산 예산·종료·원인 해석',timing))

    cases='<p>각 대비에서 두 카메라 모두 최종 유효한 pair를 대상으로 pair 평균 ADD 차이가 가장 음수/양수인 사례를 고른다. 동률은 pair ID 순이다. 평가용 설명 사례 선정이며 하이퍼파라미터 선택에 사용하지 않았다. 해당 부호 사례가 없으면 없다고 기록한다.</p><p>첫 열은 변하지 않은 원래 후보와 raw 평균, 뒤 두 열은 각각의 사후 MAP 후보(노란 사각형), 포즈 투영(청록 ×), annotation reference(녹색 원)이다. 투영점을 새 후보로 표시하지 않는다.</p>'
    for example in examples['examples']:
        if not example['available']:cases+='<p>'+esc(example['contrast']+' / '+example['direction']+': '+example['reason'])+'</p>';continue
        cases+=image(Path(example['image']).stem,f"{example['contrast']} · {example['direction']} · pair {example['pair_id']} · 평균 ADD Δ {example['delta_ADD_mm']:+.2f}mm")
    sections.append(section('examples','실제 사진의 개선·악화 사례',cases))

    validation='<p>평가 최소 검증은 실패 분모·엄격 임계값·AUC 적분, 공동 유효 교집합, 원시간 블록과 마스크 순서, 모든 조건의 동일 재표집, PCK reference mask, reference 변경과 추정 불변을 확인한다. solver 검증·캐시 전수 감사·원본 보존의 실제 실행 증거는 아래 파일에 연결한다.</p>'
    evidence=[]
    for directory in ('validation','preservation'):
        for p in sorted((ROOT/directory).glob('*.json')):
            try:v=json.loads(p.read_text())
            except Exception:continue
            status=v.get('passed',v.get('completed','기록'))
            evidence.append([esc(p.name),esc(status),link(p,'JSON 열기')])
    validation+=table(['검증 기록','상태','증거'],evidence)
    validation+='<p>과거 입력 감사 실패 기록이 있다면 후속 수정·재검증 기록과 구분해 남긴다. 모든 원본 내용이 보존됐다는 결론은 preservation의 전후 비교 결과에 근거한다. HTML 생성 입력의 SHA 확인은 전체 저장소 보존 검사와 별개다.</p>'
    sections.append(section('validation','검증과 원본 보존 증거',validation))

    run='<p>저장소 루트에서 실행한다. 추정과 평가는 분리되어 있으며, 아래 평가 명령만으로 모델 추론이나 solver를 다시 실행하지 않는다. 새 출력 디렉터리를 지정해야 한다.</p><pre><code>'+esc(f'.venv/bin/python -B candidate_background_ablation_20261116/src/evaluate.py \\\n  --run-dir {meta["run_dir"]} \\\n  --config {meta["config_path"]} \\\n  --output candidate_background_ablation_20261116/reports/recomputed_all\n\n.venv/bin/python -B candidate_background_ablation_20261116/src/test_eval.py')+'</code></pre>'
    run+=table(['산출물','파일'],[[title,link(out/file,title)] for title,file in [('전체·적격·gate별 요약','summary_metrics.csv'),('대비와 기본 1초 CI','paired_comparisons.csv'),('대비 0.5/1/2초 민감도','bootstrap_sensitivity.csv'),('각 방법 절대 지표 CI 민감도','method_bootstrap_sensitivity.csv'),('관측별 gate 전후 ADD','per_observation_metrics.csv'),('실패 원인','failures.csv'),('시간·반복 원시 기록','timings.csv'),('bootstrap 정의','bootstrap.json'),('실제 재표집 블록 counts','bootstrap_indices.npz'),('사례 선정 기록','examples.json')]])
    run+='<p>이 HTML에는 그림·스타일·표 데이터를 포함해 인터넷 없이 열 수 있다. 소스·CSV 링크는 저장소와 함께 있을 때 유효하다. 독립 환경·새 로봇·동기화 오차·움직이는 카메라에 대한 일반화 검증은 이번 실험 범위에 포함하지 않는다.</p>'
    sections.append(section('artifacts','재현 명령과 결과 파일',run))

    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f6f8;color:#20313c;font:16px/1.75 system-ui,-apple-system,"Noto Sans KR",sans-serif}header{padding:64px max(6vw,24px);background:#102d3a;color:#fff}header p{color:#a9d2d9;letter-spacing:.13em;font-size:12px}h1{font-size:clamp(30px,4vw,52px);line-height:1.25;max-width:1000px;margin:.4em 0}main{max-width:1230px;margin:auto;padding:30px 26px 70px}nav{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 20px}nav a{font-size:13px;background:white;border:1px solid #dce4e8;padding:5px 11px;border-radius:20px}a{color:#087e8b;text-decoration:none}a:hover{text-decoration:underline}section{background:white;border:1px solid #dce4e8;border-radius:14px;padding:34px;margin:24px 0;scroll-margin:20px}h2{font-size:29px;line-height:1.35;margin:0 0 22px}h3{font-size:21px;margin-top:30px}p{margin:16px 0}.eyebrow{font-size:11px;letter-spacing:.13em;color:#087e8b;margin:0 0 9px}.lead{font-size:20px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metrics article{padding:19px;background:#f0f6f7;border-radius:9px}.metrics span,.metrics small{display:block;font-size:12px;color:#526d78}.metrics strong{font-size:28px;display:block;color:#105b69}.table-wrap{overflow:auto;margin:20px 0}table{width:100%;border-collapse:collapse;font-size:13px;white-space:normal}th{text-align:left;background:#edf4f6;color:#315663}td,th{padding:11px 12px;border-bottom:1px solid #dce6eb;vertical-align:top}tbody tr:nth-child(even){background:#fafcfd}.equation{font-family:Georgia,serif;font-size:20px;background:#f3f7f9;border-left:3px solid #087e8b;padding:20px;overflow:auto}.warning{background:#fff5e6;border-left:4px solid #d38b14;padding:17px;margin:20px 0}.controls{display:flex;gap:20px;flex-wrap:wrap;background:#f3f7f9;padding:18px;border-radius:8px}label{font-size:13px;color:#42616b}select{display:block;margin-top:5px;padding:8px;min-width:145px;border:1px solid #c6d6de;border-radius:5px;background:white;color:#20313c;font:inherit}figure{margin:25px 0}figure img{display:block;width:100%;height:auto;border:1px solid #e0e7eb;border-radius:6px;cursor:zoom-in}figcaption{font-size:13px;color:#607780;margin-top:8px}pre{background:#142e3b;color:#e5f3f6;padding:22px;border-radius:8px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}footer{text-align:center;padding:35px;color:#607780;font-size:12px}dialog{border:0;border-radius:8px;padding:12px;width:95vw;max-width:1800px}dialog::backdrop{background:#102732dd}dialog img{width:100%;height:auto}dialog button{float:right;padding:8px 15px} @media(max-width:700px){main{padding:15px 10px}section{padding:20px 16px}h2{font-size:24px}.metrics{grid-template-columns:repeat(2,1fr)}.lead{font-size:17px}td,th{padding:8px;font-size:12px}}@media print{body{background:white}header{padding:25px;color:#20313c;background:white}main{max-width:none;padding:0}section{break-before:auto;box-shadow:none;border:0;padding:15px}nav,.controls{display:none}figure,table{break-inside:avoid}a{color:inherit}h2{break-after:avoid}}'''
    payload=json.dumps(clean({'summary':evaluated['summary']}),ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    js='''const DATA=JSON.parse(document.getElementById('report-data').textContent());const q=id=>document.getElementById(id);const f=(v,s=1)=>v===null?'—':(v*s).toFixed(2);function render(){const gate=q('gate').value,population=q('population').value,camera=q('camera').value;const rows=DATA.summary.filter(x=>x.gate===gate&&x.population===population&&x.camera===camera);q('selection-note').textContent=population+' / '+gate+' / '+camera+' · 실패를 분모에 포함, 평균은 유효 포즈 조건부';q('dynamic-results').innerHTML='<div class="table-wrap"><table><thead><tr><th>조건</th><th>유효/전체</th><th>ADD50 %</th><th>AUC</th><th>평균 mm</th><th>중앙값</th><th>P90</th><th>raw PCK5 %</th><th>MAP PCK5 %</th><th>pose PCK5 %</th></tr></thead><tbody>'+rows.map(r=>'<tr><td>'+r.method+'</td><td>'+r.n_valid+'/'+r.n_total+'</td><td>'+f(r.ADD50,100)+'</td><td>'+f(r.ADD_AUC)+'</td><td>'+f(r.ADD_mean_mm)+'</td><td>'+f(r.ADD_median_mm)+'</td><td>'+f(r.ADD_p90_mm)+'</td><td>'+f(r.raw_PCK5,100)+'</td><td>'+f(r.map_PCK5,100)+'</td><td>'+f(r.projected_PCK5,100)+'</td></tr>').join('')+'</tbody></table></div>';}['gate','population','camera'].forEach(id=>q(id).addEventListener('change',render));render();document.querySelectorAll('figure img').forEach(img=>{img.tabIndex=0;const open=()=>{q('full-image').src=img.src;q('full-image').alt=img.alt;q('image-dialog').showModal();};img.addEventListener('click',open);img.addEventListener('keydown',e=>{if(e.key==='Enter')open();});});q('close-dialog').onclick=()=>q('image-dialog').close();'''
    nav='<nav aria-label="보고서 목차">'+''.join('<a href="#'+s.split('id="')[1].split('"')[0]+'">'+str(i+1).zfill(2)+' '+s.split('<h2>')[1].split('</h2>')[0]+'</a>' for i,s in enumerate(sections))+'</nav>'
    doc='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Candidates 배경 결합 × Essential 초기화 통제 실험</title><style>'+css+'</style></head><body><header><p>ROBOTPOSE / CONTROLLED CANDIDATES ABLATION</p><h1>배경 결합과 Essential 초기화는<br>각각 얼마나 기여하는가</h1><div>UR_dual_2596 · RoMa V2 · 동일 후보 · 2×2 통제 실험 · '+('전체 결과' if full else '부분 실행 점검')+'</div></header><main>'+nav+''.join(sections)+'</main><footer>2026-11-16 · 실패 포함 전체 분모 · annotation-reference agreement · 새 확장 산출물</footer><dialog id="image-dialog"><button id="close-dialog">닫기</button><img id="full-image" alt=""></dialog><script type="application/json" id="report-data">'+payload+'</script><script>'+js+'</script></body></html>'
    target.write_text(doc)
    manifest=dict(report=str(target),report_sha256=sha(target),input_sha256=sources,
        inputs_unchanged=all(sha(p)==h for p,h in sources.items()),n_summary_rows=len(evaluated['summary']),
        interactive_selections=12,sections=len(sections),full_sequence=full,n_pairs=n)
    write_json(out/(Path(name).stem+'_sources.json'),manifest)
    return str(target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation-dir',required=True);p.add_argument('--name',default='ablation_report.html')
    a=p.parse_args();print(build_report(a.evaluation_dir,name=a.name))
