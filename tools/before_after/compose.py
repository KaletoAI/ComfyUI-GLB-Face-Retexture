"""Compose turning-head frame folders into one labelled before/after GIF.

Usage:
    python tools/before_after/compose.py OUT.gif --titles "Original mesh|Face retexture|..." \
        --row orig_a plain_a full_a --row orig_b plain_b full_b [--frames 24] [--size 320]

Each --row lists one frame folder per column (written by turn_head.py). Needs Pillow.
"""
import argparse

from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('--titles', required=True)
parser.add_argument('--row', nargs='+', action='append', required=True)
parser.add_argument('--frames', type=int, default=24)
parser.add_argument('--size', type=int, default=320)
args = parser.parse_args()
titles = args.titles.split('|')
size, top, background = args.size, 30, (246, 246, 246)
try:
    font = ImageFont.truetype('DejaVuSans.ttf', 14)
except OSError:
    font = ImageFont.load_default()
frames = []
for i in range(args.frames):
    canvas = Image.new('RGB', (size * len(titles), top + size * len(args.row)), background)
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(titles):
        width = draw.textlength(title, font=font)
        draw.text((column * size + (size - width) / 2, 8), title, fill=(30, 30, 30), font=font)
        for row, folders in enumerate(args.row):
            frame = Image.open(f'{folders[column]}/f{i:03d}.png').convert('RGBA')
            plate = Image.new('RGBA', frame.size, background + (255,))
            plate.alpha_composite(frame)
            canvas.paste(plate.convert('RGB').resize((size, size)), (column * size, top + row * size))
    frames.append(canvas.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)
