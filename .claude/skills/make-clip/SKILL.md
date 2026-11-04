---
name: make-clip
description: Render a pixel-style mp4 from a saved policy checkpoint or a recorded episode, including side-by-side generation comparisons. Use when an experiment produced a moment worth showing or an episode needs footage.
---

# /make-clip

학습된 정책(체크포인트)이나 기록된 에피소드를 도트 스타일 영상으로 만든다. 릴스/쇼츠 규격을 기본으로 한다.

## 입력으로 받을 것
- 실험 번호와 체크포인트(세대) 하나 또는 여러 개
- 클립 종류: `single`(한 세대 플레이), `compare`(세대 N vs M 나란히), `evolution`(여러 세대를 순서대로 이어 붙이기)
- 용도: `shorts`(9:16, 1080x1920, 15~45초) 또는 `episode`(16:9, 1920x1080)

## 절차
1. `uv run python -m render.clip --exp EXP-NNN --ckpt gen120 --kind compare --format shorts --seed 7` 형태로 `render/` 모듈을 호출한다. 없으면 만든다(Pillow 프레임 → ffmpeg).
2. 같은 시드로 여러 에피소드를 돌려 **가장 설명이 잘 되는** 플레이를 고른다. 평균적인 플레이가 아니라 차이가 보이는 플레이.
3. 화면 위에 자막 띠: 세대 번호, 짧은 설명(예: "120세대 · 사이드라인에 붙어버림"). 릴스 레퍼런스처럼 큰 도트 글꼴.
4. 결과는 `experiments/EXP-NNN/clips/`에 저장하고, 쇼츠 후보면 `content/episodes/epNN/shorts.md`에 한 줄 추가.
5. 해당 실험의 `notes.md`에 클립 경로를 적는다(`#story` 장면이면 태그와 함께).

## 규칙
- 선수는 원형 스프라이트 + 가상 팀 색. NFL 로고나 팀명은 쓰지 않는다.
- 비교 클립은 반드시 **같은 시드, 같은 초기 배치**로 만든다. 다르면 비교가 아니다.
- 대표 클립은 `experiments/EXP-NNN/clips/`에 두고 커밋한다(클립당 1MB 안팎). 다른 위치의 mp4는 커밋하지 않는다.
