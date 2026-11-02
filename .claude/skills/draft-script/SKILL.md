---
name: draft-script
description: Turn experiment notes and the devlog into a Yosh-style narrated episode script, a shot list, and shorts candidates. Use when a curriculum stage (1v1, 3v3, passing…) has enough recorded experiments to tell a story.
---

# /draft-script

레퍼런스: [Yosh — Training an unbeatable AI in Trackmania](https://youtu.be/Dw3BZ6O_8LY). 결과가 아니라 **과정의 이야기**다.
"이렇게 해봤는데 → AI가 이런 짓을 했고 → 그래서 이렇게 바꿨다"의 반복이 뼈대다.

## 입력
- 에피소드 범위(예: 1v1 단계, EXP-001~008)
- 해당 실험들의 `notes.md` 전부와 그 기간의 `devlog/`
- `#story` 태그가 붙은 장면 목록

## 산출물 (`content/episodes/epNN-이름/`)
1. `script.md` — 내레이션 대본. 사용자(1인칭, 구어체, 입문자 시점)의 말투. 8~12분 분량(한국어 기준 약 2,000~2,800자).
2. `shots.md` — 문단마다 어떤 화면을 보여줄지. 클립 경로, 그래프, 코드 화면, 얼굴 캠 여부.
3. `shorts.md` — 이 에피소드에서 15~45초로 잘라낼 수 있는 장면 3~5개와 첫 문장(훅).

## 대본 구조
1. **훅 (0:00–0:30)**: 가장 이상하거나 웃긴 장면을 먼저 보여준다. "AI한테 미식축구를 가르쳤더니 사이드라인을 껴안기 시작했습니다."
2. **설정 (0:30–2:00)**: 뭘 하려는지, 규칙은 안 가르치고 보상만 준다는 것, 미식축구 용어 1~2개 설명.
3. **순환 (본문)**: 실험 순서대로. 각 순환은 "기대 → 실제 → 왜 → 수정". 숫자(세대, 성공률)를 자막에 넣는다.
4. **전환점**: 전술이 처음 "보이기" 시작한 순간. 가장 길게 다룬다.
5. **마무리**: 배운 것 한 줄, 다음 단계 예고, 저장소 링크.

## 규칙
- 실험 노트에 없는 일은 쓰지 않는다. 과장 금지, 실패는 그대로.
- 전문 용어는 처음 나올 때 한 문장으로. 시청자도 입문자다.
- 대본은 초안이다. 사용자가 자기 말로 고쳐 녹음한다. 게시는 사용자가 한다.
- 초안이 끝나면 "첫 시청자 리뷰" 서브에이전트에게 맥락 없이 읽혀 막히는 곳을 받는다(선택).
