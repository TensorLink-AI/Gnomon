# Gnomon marketing

## The Honest Instrument — a sixty-second film

`the-honest-instrument.html` is a self-playing, zero-dependency marketing
teaser built around the product's own name: a *gnomon* is the part of the
sundial that casts the shadow — the one instrument that never guessed, and
said nothing when the light was gone. The film lands on Gnomon's actual
differentiators: backtested selection, conformal intervals, support tiers,
and honest abstention.

### Watch it

Open the file in any browser — it plays itself (60 seconds, silent).

- Click the stage or press space to pause; ←/→ jump scenes; R restarts.
- A scrubber and scene name sit under the stage.
- `prefers-reduced-motion` viewers get a paused film they can step through.

### Use it

- **Share as a link** — host the single file anywhere (GitHub Pages, S3, a
  claude.ai artifact). No build step, no external assets beyond Google Fonts.
- **Record an MP4 for socials** — play it fullscreen and screen-record the
  stage at 1080p (QuickTime, OBS, or `xdg` screencast). The stage is a strict
  16:9 letterboxed region, so the crop is clean.
- **Reuse frames as stills** — pause on any scene for launch-post imagery;
  the closing lockup doubles as a social card.

### Edit it

Everything lives in the one file:

- Scene copy is plain HTML in the `.scene` blocks.
- Timings are the `bounds` array in the script (seconds per scene, 60s total).
- The terminal scripts are `SCRIPT5` / `SCRIPT6`; the chart is drawn in
  `drawChart()` from deterministic data, so scrubbing is frame-exact.
- Palette and type are CSS variables at the top (Cinzel / Spectral /
  IBM Plex Mono).

The demo output shown in the film is illustrative copy modeled on real CLI
behavior; keep it consistent with `gnomon capabilities` when the product
changes.
