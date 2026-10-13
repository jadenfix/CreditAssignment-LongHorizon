#!/usr/bin/env python3
"""Render the verified local CTS fixture receipt as GIF and MP4."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 720
FPS = 6
COLORS = {
    "ink": "#F8FAFC",
    "muted": "#94A3B8",
    "panel": "#111C33",
    "panel2": "#17233E",
    "violet": "#8B5CF6",
    "cyan": "#22D3EE",
    "green": "#34D399",
    "red": "#FB7185",
    "amber": "#FBBF24",
}


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def _round(draw: ImageDraw.ImageDraw, box, fill, radius=22, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _text(draw, xy, value, size, fill=None, bold=False, anchor=None):
    draw.text(
        xy,
        value,
        font=_font(size, bold=bold),
        fill=fill or COLORS["ink"],
        anchor=anchor,
    )


def _base(progress: float) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#07101F")
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        blend = y / HEIGHT
        draw.line(
            (0, y, WIDTH, y), fill=(7 + int(8 * blend), 16 + int(9 * blend), 31 + int(18 * blend))
        )
    draw.ellipse((930, -250, 1460, 280), fill="#15275A")
    draw.ellipse((-260, 500, 310, 1050), fill="#10283B")
    _text(draw, (64, 46), "CRITIQUE–TRANSPORT SMOOTHING", 18, COLORS["cyan"], True)
    _text(draw, (1216, 46), "LOCAL EXECUTION RECEIPT", 16, COLORS["muted"], True, "ra")
    draw.rounded_rectangle((64, 679, 1216, 686), radius=4, fill="#22304A")
    draw.rounded_rectangle(
        (64, 679, 64 + int(1152 * progress), 686), radius=4, fill=COLORS["violet"]
    )
    return image, draw


def _badge(draw, x, y, label, color):
    width = draw.textbbox((0, 0), label, font=_font(15, bold=True))[2] + 30
    _round(draw, (x, y, x + width, y + 34), color, radius=17)
    _text(draw, (x + width / 2, y + 17), label, 15, "#07101F", True, "mm")
    return x + width


def _metric(draw, x, y, title, value, accent):
    _round(draw, (x, y, x + 250, y + 112), COLORS["panel"], outline="#283855")
    _text(draw, (x + 22, y + 20), title.upper(), 14, COLORS["muted"], True)
    _text(draw, (x + 22, y + 55), value, 30, accent, True)


def render_frame(receipt: dict, logs: list[dict], frame: int, total: int) -> Image.Image:
    progress = frame / max(total - 1, 1)
    image, draw = _base(progress)
    phase = min(4, int(progress * 5))
    fade = 0.5 - 0.5 * math.cos(min(1.0, (progress * 5) % 1.0) * math.pi)
    rollout = receipt["fixture_rollout"]

    if phase == 0:
        _text(draw, (64, 155), "A real CTS optimizer step,", 48, COLORS["ink"], True)
        _text(draw, (64, 215), "small enough to audit.", 48, COLORS["violet"], True)
        _text(
            draw,
            (66, 304),
            "Replay-only data  +  fresh nano-LM samples  +  verifier rewards",
            23,
            COLORS["muted"],
        )
        x = _badge(draw, 66, 365, "CPU", COLORS["cyan"]) + 12
        x = _badge(draw, x, 365, f"{receipt['steps']} OPTAX STEPS", COLORS["green"]) + 12
        _badge(draw, x, 365, "NO DOWNLOADS", COLORS["amber"])
        _round(draw, (64, 448, 1216, 612), COLORS["panel"], outline="#283855")
        _text(draw, (92, 478), "TINY RANDOM FIXTURE", 22, COLORS["amber"], True)
        _text(draw, (92, 521), "No quality or scale claim", 34, COLORS["ink"], True)
        _text(
            draw,
            (92, 570),
            "The receipt proves execution and accounting—not research performance.",
            19,
            COLORS["muted"],
        )
    elif phase == 1:
        _text(
            draw,
            (64, 112),
            "One comparison group. Three traceable sources.",
            38,
            COLORS["ink"],
            True,
        )
        labels = [
            ("ARCHIVED y0", "flawed replay attempt", "0", COLORS["red"]),
            ("ARCHIVED y1", "corrected replay revision", "1", COLORS["green"]),
            ("NANO-LM", "fresh sampled rollout", "0", COLORS["violet"]),
        ]
        for idx, (title, subtitle, reward, accent) in enumerate(labels):
            x = 64 + idx * 392
            y = 220 + int(8 * (1 - fade))
            _round(draw, (x, y, x + 360, y + 250), COLORS["panel"], outline="#30415F")
            _text(draw, (x + 24, y + 26), title, 18, accent, True)
            _text(draw, (x + 24, y + 65), subtitle, 17, COLORS["muted"])
            _text(draw, (x + 24, y + 124), "VERIFIER REWARD", 13, COLORS["muted"], True)
            _text(draw, (x + 24, y + 150), reward, 58, accent, True)
            draw.rounded_rectangle((x + 108, y + 171, x + 326, y + 185), radius=7, fill="#24314A")
            if reward == "1":
                draw.rounded_rectangle((x + 108, y + 171, x + 326, y + 185), radius=7, fill=accent)
        _text(draw, (64, 520), f"reward builder  {rollout['reward_id']}", 17, COLORS["cyan"], True)
        _text(
            draw,
            (64, 552),
            f"candidate tensor  {rollout['candidate_group_shape']}    sampled tensor  {rollout['sampled_rollout_shape']}",
            19,
            COLORS["muted"],
        )
    elif phase == 2:
        _text(draw, (64, 112), "The optimizer moved real parameters.", 40, COLORS["ink"], True)
        final_log = logs[-1]
        _metric(draw, 64, 210, "finite loss", f"{receipt['loss']:.4f}", COLORS["green"])
        _metric(
            draw, 334, 210, "parameter Δ L1", f"{receipt['parameter_delta_l1']:.4f}", COLORS["cyan"]
        )
        _metric(draw, 604, 210, "reward mean", f"{rollout['reward_mean']:.3f}", COLORS["amber"])
        _metric(
            draw,
            874,
            210,
            "generated shape",
            str(receipt["generated_token_shape"]),
            COLORS["violet"],
        )
        _round(draw, (64, 362, 1216, 590), COLORS["panel"], outline="#283855")
        _text(draw, (92, 390), "ACTUAL TRAIN LOG", 14, COLORS["muted"], True)
        for idx, item in enumerate(logs):
            y = 438 + idx * 60
            _text(draw, (92, y), f"step {item['step']}", 18, COLORS["cyan"], True)
            _text(draw, (208, y), f"loss {item['loss']:.4f}", 18)
            _text(draw, (405, y), f"GRPO {item['grpo_loss']:.4f}", 18, COLORS["muted"])
            _text(draw, (625, y), f"decode {item['decode_anchor']:.4f}", 18, COLORS["muted"])
        _text(draw, (920, 500), f"final total  {final_log['total']:.4f}", 21, COLORS["green"], True)
    elif phase == 3:
        _text(draw, (64, 112), "Fairness settings travel with the result.", 40, COLORS["ink"], True)
        fp = receipt["fairness_fingerprint"]
        rows = [
            ("prompt template", fp["prompt_template_hash"]),
            ("optimizer steps", str(fp["total_opt_steps"])),
            ("decode temperature", str(fp["decode_temperature"])),
            ("revision budget", str(fp["revision_budget"])),
            ("replay content hash", receipt["replay_hash"]),
        ]
        _round(draw, (64, 190, 1216, 590), COLORS["panel"], outline="#283855")
        for idx, (key, value) in enumerate(rows):
            y = 226 + idx * 65
            _text(draw, (96, y), key.upper(), 15, COLORS["muted"], True)
            _text(draw, (470, y), value, 22, COLORS["ink"], True)
            if idx < len(rows) - 1:
                draw.line((96, y + 43, 1184, y + 43), fill="#263650", width=1)
        _badge(draw, 944, 530, "RECEIPT VERIFIED", COLORS["green"])
    else:
        _text(draw, (64, 128), "Verified local workflow", 45, COLORS["ink"], True)
        _text(draw, (64, 188), "from replay to rollout to update.", 45, COLORS["violet"], True)
        _round(draw, (64, 292, 1216, 532), COLORS["panel"], outline="#31415F")
        checks = [
            "fresh nano-LM samples generated",
            "fixture verifier rewards recorded",
            "CTS + GRPO loss remained finite",
            "model and auxiliary parameters changed",
        ]
        for idx, check in enumerate(checks):
            y = 328 + idx * 47
            draw.ellipse((96, y, 116, y + 20), fill=COLORS["green"])
            _text(draw, (109, y + 10), "✓", 14, "#07101F", True, "mm")
            _text(draw, (138, y - 1), check, 20)
        _text(
            draw,
            (64, 584),
            "TINY RANDOM FIXTURE  /  NO QUALITY OR SCALE CLAIM",
            19,
            COLORS["amber"],
            True,
        )
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--gif", type=Path, required=True)
    parser.add_argument("--mp4", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    logs = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]
    total_frames = 60
    with tempfile.TemporaryDirectory(prefix="cts-demo-") as directory:
        frame_dir = Path(directory)
        for frame in range(total_frames):
            render_frame(receipt, logs, frame, total_frames).save(frame_dir / f"{frame:04d}.png")
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(FPS),
                "-i",
                str(frame_dir / "%04d.png"),
                "-vf",
                "fps=6,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer",
                "-loop",
                "0",
                str(args.gif),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(FPS),
                "-i",
                str(frame_dir / "%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-crf",
                "22",
                str(args.mp4),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
