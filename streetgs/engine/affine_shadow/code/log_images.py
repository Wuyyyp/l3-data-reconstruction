"""Diagnostic images from the existing forward pass; no extra rendering."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def save_log_image(output, iteration, camera_name, gt, raw, affine):
    panels = []
    for tensor in (gt, raw, affine):
        array = tensor.detach().float().cpu().numpy()
        image = Image.fromarray((np.clip(array, 0, 1).transpose(1, 2, 0) * 255).round().astype('uint8'))
        width = min(960, image.width)
        image = image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
        panels.append(image)
    width, height = panels[0].size
    assert all(p.size == (width, height) for p in panels)
    canvas = Image.new('RGB', (3 * width, height + 40), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), 'Step %d | %s' % (iteration, camera_name), fill='black')
    for index, (label, panel) in enumerate(zip(('GT', 'Raw', 'Affine'), panels)):
        draw.text((index * width + 8, 23), label, fill='black')
        canvas.paste(panel, (index * width, 40))
    directory = Path(output) / 'log_images'
    directory.mkdir(exist_ok=True)
    target = directory / ('%d.jpg' % iteration)
    temp = target.with_suffix('.jpg.tmp')
    canvas.save(temp, format='JPEG', quality=90)
    temp.replace(target)
