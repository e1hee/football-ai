---
name: log-experiment
description: Record a finished experiment in its experiments/EXP-NNN folder and the daily devlog. Use right after any training run or sim change finishes, even a failed one.
---

# /log-experiment

실험이 끝난 직후, 실험을 돌린 바로 그 세션이 실행한다. 기억이 생생할 때만 쓸 수 있는 내용이 있다.

## 절차
1. 실험 번호를 정한다. `experiments/`에서 가장 큰 번호 + 1. 폴더명은 `EXP-NNN-짧은이름` (예: `EXP-003-1v1-sideline-penalty`).
2. 폴더에 `config.yaml`(실제 사용한 설정, 시드 포함)과 지표 파일(`metrics.json` 또는 csv)을 둔다. 대표 클립이 있으면 경로를 적는다(파일은 커밋 안 함).
3. `notes.md`를 아래 양식으로 쓴다. 다섯 항목 모두 채운다. 숫자는 반드시 넣는다.
4. `devlog/YYYY-MM-DD.md`에 한 줄 추가: `- EXP-NNN 이름 — 결과 한 줄`.
5. 커밋: `EXP-NNN: 한 줄 요약`.

## notes.md 양식

```markdown
# EXP-NNN 이름
날짜: YYYY-MM-DD · 기반: EXP-MMM (없으면 "없음") · 학습 시간: Xh · 스텝: N

## 가설
무엇이 왜 될 거라고 생각했나. 한두 문장.

## 시도
무엇을 바꿨나. 설정 차이만 구체적으로.

## 결과
숫자로. 성공률, 평균 야드, 세대 간 승률, 학습 곡선 파일 경로.

## 놀라운 점
예상과 다른 것. 꼼수, 이상한 행동, 생각보다 잘/못 된 것. 없으면 "없음"이라고 쓴다.
#story 태그가 붙을 만한 장면은 클립 경로와 함께: `#story clips/gen120_sideline_hug.mp4 — 러너가 사이드라인에 붙어 무한 대기`

## 다음 시도
이 결과 때문에 다음에 뭘 바꿀 건가.
```

## 하지 말 것
- "잘 됐다/안 됐다"만 쓰기. 숫자 없는 결과는 결과가 아니다.
- 놀라운 점을 비워두고 넘어가기. 영상은 여기서 나온다.
- 여러 실험을 하나의 notes.md에 몰아 쓰기.
