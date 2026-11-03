# Football AI

Teaching AI American football with **rewards only**: no plays, no rules of thumb, just "gain yards".
A vectorized 2D sim with movement limits fitted to real NFL tracking data, PPO, and self-play.
The whole process is logged experiment by experiment for a narrated video series.

> 보상만으로 AI에게 미식축구를 가르치고, 그 과정을 실험 하나하나 기록해 영상으로 만드는 프로젝트.

<p align="center"><img src="experiments/EXP-004-fine-gens-runner/clips/exp004_runner_grid.png" width="540"></p>
<p align="center"><sub>Same 9 starting setups, 61 generations of the AI runner drawn at once (purple = generation 0, yellow = generation 60).</sub></p>

## Experiments so far

| # | What | Result |
|---|---|---|
| [EXP-000](experiments/EXP-000-baseline-scripted/notes.md) | Bare 1v1 sim (runner vs tackler), scripted policies | Running straight scores 80%: discs pass through each other, no physics |
| [EXP-001](experiments/EXP-001-physics-and-tackling/notes.md) | NFL-fitted acceleration/braking/cutting, collisions (checked against Pymunk), probabilistic tackles | With momentum, a chasing defender overruns the runner. Taking an angle and breaking down drops straight-run TDs to 29% |
| [EXP-002](experiments/EXP-002-ppo-runner-v1/notes.md) | PPO runner vs the scripted tackler | 0% → 99% touchdowns in 2 generations (4 minutes). It found a hole in the tackler's prediction rule, not football |
| [EXP-003](experiments/EXP-003-selfplay-v1/notes.md) | Self-play: the tackler learns too | The champion runner drops 100% → 61% in one phase, then both sides stall near 60% (the 1v1 equilibrium). The tackler learned to wait in the runner's lane instead of chasing |
| [EXP-004](experiments/EXP-004-fine-gens-runner/notes.md) | Retrain with a checkpoint every 100k steps; draw all 61 generations at once on a colour gradient (`render/ghosts.py`) | The runner switched strategies mid-evolution (right side in gens 38-51, left from gen 52). Tackler that can't make the play just gives up and drifts to a corner |
| [EXP-005](experiments/EXP-005-delay-control/notes.md) | 0.2 s reaction delay + tackle control (overspeed / sidestep), compared to NFL pre-contact movement | Runner cuts reach NFL level (17-21° vs 19°) but nobody brakes. The tackler's learning collapsed: in ~1/3 of plays it runs away from the runner from the first step |
| [EXP-006](experiments/EXP-006-joystick-fix/notes.md) | Fix actions collapsing to 8 directions (SB3 clips each axis; widened the box) | Failed: tacklers learned to flee downfield 100% of the time, exploiting a loose breakaway rule against a runner that barely moved |
| [EXP-007](experiments/EXP-007-fixed-selfplay/notes.md) | Stricter breakaway rule, no untrained opponents in pools, plus a hand-written read-and-react tackler | First real arms race: the cross table forms a triangle (new generations beat old ones on both sides). Tacklers learned to pin runners to the sideline |
| [EXP-008](experiments/EXP-008-facing/notes.md) | Body facing (hips) measured from NFL tracking `o`: limited turn rate, slower moving away from where you face | First pre-contact braking: tacklers now brake before contact (median 2.5 yd/s², NFL 4.0). Runners still don't juke in open-field 1v1 |
| [EXP-009](experiments/EXP-009-3v3-screen/notes.md) | First 3v3: RB screen (RB + 2 OL vs LB + 2 DB) with blocking, OL speed calibrated to the combine | Defense wins the arms race (latest-vs-latest TD 84% → 42%), tacklers brake like NFL defenders (median 3.9 vs 4.0), one DB learns to play deep safety on its own |
| [EXP-010](experiments/EXP-010-blocker-shaping/notes.md) | Role reward for the blockers (potential-based shaping) | Failed: a bug in my shaping paid a bonus at timeouts, and the offense learned to send its blockers to the corners and run out the clock (71% timeouts) |
| [EXP-011](experiments/EXP-011-blocker-shaping-fixed/notes.md) | Same role reward, timeout bug fixed | Looked like fewer lost blockers (6% → 1%), but EXP-012 showed that was one lucky seed; the offense is no stronger |
| [EXP-012](experiments/EXP-012-seed-repeats/notes.md) | Three seeds each for EXP-009 and EXP-011, every offense vs every defense | The role reward does nothing (offense TD 56.1% vs 55.7%); seed luck is far bigger (defenses allow 38% to 76% under the same settings) |
| [EXP-013](experiments/EXP-013-lane-drill/notes.md) | Oklahoma lane drill (RB + OL vs DL in a 5-yd lane, engaged at the snap; an engaged DL can tackle on his free side), 3 seeds | The runner learns to read the block in 2 generations (hits the open side ~90% vs 30% for a runner who picks at the snap), but still never slows down to wait; offense too strong (72% AI vs AI) |
| [EXP-014](experiments/EXP-014-lane-lb/notes.md) | Same drill 2v2: add an unblocked LB | Defense wins big (runner 9%, 0.8 yd; the AI LB beats the runner to the hole). Runners never brake on their own: in the 2v1 drill 98% of their pre-line slowdown came from running into their own blocker's back, a free super-brake in the collision physics |
| [EXP-015](experiments/EXP-015-balance/notes.md) | Hard collisions knock you off balance; the runner squares up only to defenders ahead | Hips fixed (runners no longer look back after breaking free: 14-53% → 0-1%), offense 9% → 18%, but self-braking only 1% → 3-4%: in this drill waiting never helps, because no block gets better with time |
| [EXP-016](experiments/EXP-016-combo/notes.md) | Combo-block drill (RB + 2 OL vs DL + LB): double teams, blockers can let go to climb to the LB | First try failed: the runner hovered 2 yd short of the end line until the clock ran out (timeouts were bootstrapped). With the clock as a real end: the most balanced drill so far (48% AI vs AI, 5.4 yd); one of three seeds learned the double team; still no waiting |
| [EXP-017](experiments/EXP-017-box/notes.md) | 6v6 inside run: RB + 5 OL vs a 4-3 front (4 DL + 2 LB), one defender always unblocked | Rock-paper-scissors: each seed's defense stops its own offense (6-15%) but cross-seed results range 8-93%. Reading the hole pays (scripted 42% vs 6%); self-braking reaches 14% in one matchup, 5% on average |
| [EXP-018](experiments/EXP-018-snap/notes.md) | Snap from the centre (0.6 s, RB 7 yd deep) and blocks that hold (2.5 s: 37 -> 69%) | The line holds and the runner reaches it at 2.6 s (NFL about 2.5-3.1), but all three seeds learned a sweep around the right edge: inside the tackles fell from 57% to 7% |

Each experiment folder has `notes.md` (hypothesis → attempt → result → surprises → next), `config.yaml`, metrics, and charts.
Daily log: [`devlog/`](devlog/). Plan and milestones: [`PLAN.md`](PLAN.md).

## How it works

- `sim/`: numpy sim batched over thousands of plays (~1M steps/s). Players are discs with speed-dependent push limits
  fitted from NFL Big Data Bowl 2027 tracking (`analysis/physics_from_bdb.py`; only aggregate numbers are stored).
- `train/`: Stable-Baselines3 PPO through a custom VecEnv (~100k training steps/s on a MacBook M1 Max CPU), self-play with opponent pools.
- `render/`: pixel-style MP4 clips (shorts 9:16 and episode 16:9) and generation-by-generation comparisons.

```bash
uv sync
uv run pytest
uv run python -m scripts.bench_env --all
uv run python -m train.train_runner --exp EXP-xxx --timesteps 10_000_000 --gens 10
uv run python -m render.clip --runner evasive --tackler angle --spread 8 --format shorts --out render/out/clip.mp4
```

## Honest limitations

- No running-back tracking exists in the data, so the ball carrier uses a wide receiver's body.
- Tackle success odds are an assumption (no contact data), currently missing more often than the real NFL.
- 1v1 on an open field is a toy. 3v3 with blockers comes next.

Data: NFL Big Data Bowl 2027 (CC BY-NC 4.0). The raw data is never committed. Team names and colors in the sim are fictional.
