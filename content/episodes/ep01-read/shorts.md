# 쇼츠 후보 — AI는 왜 빈 쪽으로 안 뛸까

| # | 길이 | 클립 | 첫 문장(훅) | 끝 |
|---|---|---|---|---|
| 1 | 20초 | `experiments/EXP-030-bounded/clips/A_s2_right_vs_flipped_left.mp4` | "왼쪽이 제 AI, 오른쪽은 같은 AI를 좌우로만 뒤집은 겁니다." | "32번 중 그대로 10번, 뒤집으면 24번 터치다운. 반대쪽이 비었는데 안 갑니다." |
| 2 | 25초 | `experiments/EXP-023-audit/clips/s2_runner_own_def_vs_fresh_def.mp4` | "터치다운 98% 러너가 처음 보는 수비를 만나면?" | 뒤로 물러나 7초 서 있다 시간 초과. "강한 게 아니라 익숙했던 겁니다." |
| 3 | 35초 | `experiments/EXP-034-overload/clips/A_s1_reads_flood_vs_own_side.mp4` | "네 번 실패한 걸, 수비를 바꾸자 1세대 만에 해냈습니다." | 위 터치다운 / 아래 태클. "못 읽는 게 아니라 읽을 이유가 없었습니다." |
| 4 | 30초 | `experiments/EXP-036-bait/clips/A_s0_lopsided_defense_mirrored.mp4` | "AI 수비한테 속이는 법을 가르치려다 이걸 만들었습니다." | 뒤집은 쪽 텅 빔. "제 실수였습니다." (고친 뒤 결과 한 줄 자막) |
| 5 | 20초 | `experiments/EXP-022-long/clips/s2_runner_g18_vs_g36.mp4` | "21세대 54%, 24세대 97%. 세 세대 만에 AI가 폭발했습니다." | 왼쪽 사이드라인 3.6초 터치다운. 다음 쇼츠(2번)로 이어지는 예고 |

- 2번과 5번은 이어서 올리면 "폭발 → 신참에게 0%" 두 편이 된다.
- 세로 출력은 `render/`의 쇼츠 레이아웃(1080x1920)으로 다시 뽑아야 한다. 지금 클립은 가로다.
