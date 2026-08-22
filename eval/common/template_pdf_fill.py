from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ArrayObject, ContentStream, DecodedStreamObject, NameObject, NumberObject, TextStringObject


EVAL_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = EVAL_ROOT / "template"
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]


def _as_rgb(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, "white")
        alpha = image.getchannel("A") if "A" in image.getbands() else None
        background.paste(image.convert("RGB"), mask=alpha)
        return background
    return image.convert("RGB")


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _summary_image(width_pt: float, height_pt: float, lines: list[str], *, font_size: int) -> Image.Image:
    scale = 4
    width_px = max(8, int(round(width_pt * scale)))
    height_px = max(8, int(round(height_pt * scale)))
    image = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(image)
    font = _load_font(font_size * scale)
    clean_lines = [str(line) for line in lines if str(line).strip()]
    if not clean_lines:
        return image

    spacing = max(4, font_size * scale // 5)
    text_boxes = [draw.textbbox((0, 0), line, font=font) for line in clean_lines]
    heights = [box[3] - box[1] for box in text_boxes]
    widths = [box[2] - box[0] for box in text_boxes]
    total_height = sum(heights) + spacing * max(0, len(clean_lines) - 1)
    y = max(0, (height_px - total_height) // 2)
    for line, line_width, line_height in zip(clean_lines, widths, heights):
        x = max(0, (width_px - line_width) // 2)
        draw.text((x, y), line, fill="black", font=font)
        y += line_height + spacing
    return image


def _write_content_stream(page, content: ContentStream) -> None:
    stream = DecodedStreamObject()
    stream.set_data(content.get_data())
    page[NameObject('/Contents')] = stream


def _remove_xobject_draws(page, reader: PdfReader, preserve_names: set[str] | None = None) -> None:
    resources = page.get('/Resources')
    if resources is None or '/XObject' not in resources:
        return
    preserve_names = preserve_names or set()
    xobject_names = set(resources['/XObject'].get_object().keys())
    removable_names = xobject_names - preserve_names
    if not removable_names:
        return
    content = ContentStream(page.get_contents(), reader)
    content.operations = [
        (operands, op)
        for operands, op in content.operations
        if not (op == b'Do' and operands and operands[0] in removable_names)
    ]
    stream = DecodedStreamObject()
    stream.set_data(content.get_data())
    page[NameObject('/Contents')] = stream


def _remove_page_background(page, reader: PdfReader) -> None:
    content = ContentStream(page.get_contents(), reader)
    ops = content.operations
    if len(ops) >= 6:
        pattern = [
            (b'BMC', ['/Artifact']),
            (b'gs', ['/GS5']),
            (b'g', [1]),
            (b're', [0, 0, 960, 540]),
            (b'f*', []),
            (b'EMC', []),
        ]
        matched = True
        for (op_expected, operands_expected), (operands, op) in zip(pattern, ops[:6]):
            if op != op_expected or list(operands) != operands_expected:
                matched = False
                break
        if matched:
            content.operations = ops[6:]
            stream = DecodedStreamObject()
            stream.set_data(content.get_data())
            page[NameObject('/Contents')] = stream

def _replace_template_figures(
    template_path: Path,
    output_pdf_path: Path,
    placements: Iterable[tuple[Path | Image.Image, tuple[float, float, float, float]]],
    *,
    rewrite_hook=None,
    align: str = 'center',
    preserve_xobject_names: set[str] | None = None,
) -> None:
    reader = PdfReader(str(template_path))
    page = reader.pages[0]
    _remove_xobject_draws(page, reader, preserve_names=preserve_xobject_names)
    _remove_page_background(page, reader)
    if rewrite_hook is not None:
        rewrite_hook(page, reader)

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for index, (image_source, box) in enumerate(placements):
            if isinstance(image_source, Path):
                if not image_source.exists():
                    continue
                image = _as_rgb(image_source)
            else:
                image = image_source.convert('RGB')
            temp_pdf = tmpdir_path / f'panel_{index}.pdf'
            image.save(temp_pdf, 'PDF', resolution=72.0)
            src_page = PdfReader(str(temp_pdf)).pages[0]
            src_w = float(src_page.mediabox.width)
            src_h = float(src_page.mediabox.height)
            x, y, w, h = box
            if src_w <= 0.0 or src_h <= 0.0 or w <= 0.0 or h <= 0.0:
                continue
            scale = min(w / src_w, h / src_h)
            if align == 'right':
                tx = x + w - src_w * scale
            elif align == 'left':
                tx = x
            else:
                tx = x + (w - src_w * scale) / 2.0
            ty = y + (h - src_h * scale) / 2.0
            transform = Transformation().scale(scale).translate(tx=tx, ty=ty)
            page.merge_transformed_page(src_page, transform, over=False)

    writer = PdfWriter()
    writer.add_page(page)
    with output_pdf_path.open('wb') as handle:
        writer.write(handle)


def _remove_sampling_training_bars(page, reader: PdfReader) -> None:
    content = ContentStream(page.get_contents(), reader)
    ops = content.operations
    filtered = []
    i = 0
    while i < len(ops):
        operands, op = ops[i]
        if op == b're' and len(operands) == 4:
            x, y, w, h = (float(operands[0]), float(operands[1]), float(operands[2]), float(operands[3]))
            if 15.0 <= w <= 25.0 and 90.0 <= h <= 110.0 and 210.0 <= y <= 230.0 and x >= 80.0:
                tail = ops[i + 1 : i + 3]
                if len(tail) >= 1 and tail[0][1] in {b'f*', b'f', b'B', b'B*', b'b', b'b*'}:
                    i += 2
                    continue
        filtered.append((operands, op))
        i += 1
    if len(filtered) != len(ops):
        content.operations = filtered
        stream = DecodedStreamObject()
        stream.set_data(content.get_data())
        page[NameObject('/Contents')] = stream


def _remove_sampling_training_text(page, reader: PdfReader) -> None:
    content = ContentStream(page.get_contents(), reader)
    filtered = []
    last_tm = None
    changed = False
    for operands, op in content.operations:
        if op == b'Tm':
            last_tm = operands
            filtered.append((operands, op))
            continue
        if op in {b'TJ', b'Tj'} and last_tm is not None:
            text = ''
            if op == b'TJ':
                for item in operands[0]:
                    if isinstance(item, str):
                        text += item
            else:
                if isinstance(operands[0], str):
                    text = operands[0]
            if text in {'Self', 'Improvemen', 't', '-'}:
                changed = True
                continue
        filtered.append((operands, op))
    if changed:
        content.operations = filtered
        stream = DecodedStreamObject()
        stream.set_data(content.get_data())
        page[NameObject('/Contents')] = stream


def _remove_memory_annotations(page, reader: PdfReader) -> None:
    content = ContentStream(page.get_contents(), reader)
    filtered = []
    skip_depth = 0
    for operands, op in content.operations:
        if skip_depth > 0:
            if op == b'BDC':
                skip_depth += 1
            elif op == b'EMC':
                skip_depth -= 1
            continue
        if op == b'BDC' and len(operands) >= 2:
            meta = operands[1]
            mcid = None
            try:
                mcid = meta.get('/MCID')
            except Exception:
                mcid = None
            if mcid in {35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51}:
                skip_depth = 1
                continue
        filtered.append((operands, op))
    content.operations = filtered
    stream = DecodedStreamObject()
    stream.set_data(content.get_data())
    page[NameObject('/Contents')] = stream


def _rewrite_memory_summary_text(page, reader: PdfReader, summary_stats: list[dict[str, float | None]] | None) -> None:
    if not summary_stats:
        return
    content = ContentStream(page.get_contents(), reader)
    panel_ops: dict[str, list[int]] = {'a': [], 'b': [], 'c': [], 'd': []}
    last_tm = None
    for index, (operands, op) in enumerate(content.operations):
        if op == b'Tm':
            last_tm = operands
            continue
        if op != b'TJ' or last_tm is None:
            continue
        try:
            x_value = float(last_tm[4])
            y_value = float(last_tm[5])
        except Exception:
            continue
        if abs(y_value - 259.2) < 0.05:
            panel_key = 'a'
        elif abs(y_value - 28.248) < 0.05:
            if x_value < 300.0:
                panel_key = 'b'
            elif x_value < 700.0:
                panel_key = 'c'
            else:
                panel_key = 'd'
        else:
            continue
        panel_ops[panel_key].append(index)

    def value_text(stats: dict[str, float | None]) -> tuple[str, str, str]:
        baseline = stats.get('others_average') if isinstance(stats, dict) else None
        ours = stats.get('ours_average') if isinstance(stats, dict) else None
        improvement = stats.get('absolute_improvement_percent') if isinstance(stats, dict) else None
        if baseline is None or ours is None or improvement is None:
            return 'nan', 'nan', 'nan'
        return f'{baseline:.2f}', f'{ours:.2f}', f'{improvement:.2f}'

    def set_tj(op_index: int, text_value: str) -> None:
        content.operations[op_index] = ([ArrayObject([TextStringObject(text_value)])], b'TJ')

    if panel_ops['a']:
        stats = summary_stats[0] if len(summary_stats) > 0 else {}
        baseline_text, ours_text, improvement_text = value_text(stats)
        replacement = f'Baselines / VLASelect avg. memory (GB): {baseline_text} / {ours_text} ({improvement_text}%)'
        set_tj(panel_ops['a'][0], replacement)
        for op_index in panel_ops['a'][1:]:
            set_tj(op_index, '')

    if len(panel_ops['b']) >= 6:
        op_indices = panel_ops['b']
        set_tj(op_indices[0], 'nan /')
        set_tj(op_indices[1], 'nan')
        set_tj(op_indices[2], '')
        set_tj(op_indices[3], '(nan%')
        set_tj(op_indices[4], ';')
        set_tj(op_indices[5], ')')
    if len(panel_ops['c']) >= 3:
        op_indices = panel_ops['c']
        set_tj(op_indices[0], 'nan / nan (nan%')
        set_tj(op_indices[1], ';')
        set_tj(op_indices[2], ')')
    if len(panel_ops['d']) >= 4:
        op_indices = panel_ops['d']
        set_tj(op_indices[0], 'nan')
        set_tj(op_indices[1], '')
        set_tj(op_indices[2], '/ nan (nan%')
        set_tj(op_indices[3], ';')
        if len(op_indices) >= 5:
            set_tj(op_indices[4], ')')

    stream = DecodedStreamObject()
    stream.set_data(content.get_data())
    page[NameObject('/Contents')] = stream
def _rewrite_accuracy_summary_text(page, reader: PdfReader, summary_stats: list[dict[str, float | None]] | None) -> None:
    if not summary_stats:
        return
    content = ContentStream(page.get_contents(), reader)
    panel_ops: dict[str, list[int]] = {"a": [], "b": [], "c": [], "d": []}
    last_tm = None
    last_tf = None
    for index, (operands, op) in enumerate(content.operations):
        if op == b'Tm':
            last_tm = operands
            continue
        if op == b'Tf':
            last_tf = operands
            continue
        if op != b'TJ' or not last_tm or float(last_tm[5]) != 234.94:
            continue
        x_value = float(last_tm[4])
        if x_value >= 700:
            panel_key = 'd'
        elif x_value >= 500:
            panel_key = 'c'
        elif x_value >= 250:
            panel_key = 'b'
        else:
            panel_key = 'a'
        panel_ops[panel_key].append(index)

    stats_by_panel = {}
    panel_order = ['a', 'b', 'c', 'd']
    for key, stats in zip(panel_order, summary_stats):
        stats_by_panel[key] = stats

    for panel_key, op_indices in panel_ops.items():
        if not op_indices:
            continue
        stats = stats_by_panel.get(panel_key, {})
        baseline = stats.get('others_average') if isinstance(stats, dict) else None
        ours = stats.get('ours_average') if isinstance(stats, dict) else None
        improvement = stats.get('absolute_improvement_percent') if isinstance(stats, dict) else None
        if baseline is None or ours is None or improvement is None:
            main_text = 'NaN / NaN (NaN)'
            keep_arrow = False
        else:
            main_text = f'{baseline:.4f} / {ours:.4f} ({improvement:.2f}%'
            keep_arrow = True

        arrow_index = None
        close_index = None
        for op_index in op_indices:
            operands, _ = content.operations[op_index]
            text_value = ''.join(item for item in operands[0] if isinstance(item, str))
            if text_value == '\x019':
                arrow_index = op_index
            elif text_value == ')':
                close_index = op_index

        first_index = op_indices[0]
        content.operations[first_index] = ([ArrayObject([TextStringObject(main_text)])], b'TJ')
        for op_index in op_indices[1:]:
            if keep_arrow and op_index == arrow_index:
                continue
            if keep_arrow and op_index == close_index:
                continue
            content.operations[op_index] = ([ArrayObject([TextStringObject('')])], b'TJ')
        if not keep_arrow and arrow_index is not None:
            content.operations[arrow_index] = ([ArrayObject([TextStringObject('')])], b'TJ')
        if not keep_arrow and close_index is not None:
            content.operations[close_index] = ([ArrayObject([TextStringObject('')])], b'TJ')

    stream = DecodedStreamObject()
    stream.set_data(content.get_data())
    page[NameObject('/Contents')] = stream


def fill_accuracy_template(output_pdf_path: Path, panel_paths: list[Path], summary_stats: list[dict[str, float | None]] | None = None) -> None:
    boxes = [
        (8.52, 272.4, 242.04, 195.6),
        (237.96, 272.4, 242.04, 195.6),
        (467.4, 272.4, 242.04, 195.6),
        (696.84, 272.4, 242.16, 195.6),
    ]
    placements: list[tuple[Path | Image.Image, tuple[float, float, float, float]]] = list(zip(panel_paths, boxes))
    _replace_template_figures(
        TEMPLATE_ROOT / "Accuracy.pdf",
        output_pdf_path,
        placements,
        rewrite_hook=(lambda page, reader: _rewrite_accuracy_summary_text(page, reader, summary_stats)),
    )


def _rewrite_resource_summary_text(page, reader: PdfReader, summary_stats: list[dict[str, float | None]] | None) -> None:
    if not summary_stats:
        return
    content = ContentStream(page.get_contents(), reader)
    panel_ops: dict[str, list[int]] = {'a': [], 'b': [], 'c': [], 'd': []}
    last_tm = None
    for index, (operands, op) in enumerate(content.operations):
        if op == b'Tm':
            last_tm = operands
            continue
        if op != b'TJ' or last_tm is None:
            continue
        try:
            x_value = float(last_tm[4])
            y_value = float(last_tm[5])
        except Exception:
            continue
        if abs(y_value - 255.82) < 0.2:
            panel_key = 'a' if x_value < 450.0 else 'b'
        elif abs(y_value - 6.1) < 0.2:
            panel_key = 'c' if x_value < 450.0 else 'd'
        else:
            continue
        panel_ops[panel_key].append(index)

    stats_by_panel = {}
    for key, stats in zip(['a', 'b', 'c', 'd'], summary_stats):
        stats_by_panel[key] = stats

    def set_tj(op_index: int, text_value: str) -> None:
        content.operations[op_index] = ([ArrayObject([TextStringObject(text_value)])], b'TJ')

    for panel_key, op_indices in panel_ops.items():
        if not op_indices:
            continue
        stats = stats_by_panel.get(panel_key, {})
        baseline = stats.get('others_average') if isinstance(stats, dict) else None
        ours = stats.get('ours_average') if isinstance(stats, dict) else None
        improvement = stats.get('absolute_improvement_percent') if isinstance(stats, dict) else None
        if baseline is None or ours is None or improvement is None:
            values = ['Baselines /', 'VLASelect', 'avg. acc: nan /', 'nan (nan%', '', ')']
        else:
            values = [
                'Baselines /',
                'VLASelect',
                f'avg. acc: {baseline:.4f} /',
                f'{ours:.4f} ({improvement:.2f}%',
                '\x019',
                ')',
            ]
        for op_index, value in zip(op_indices, values):
            set_tj(op_index, value)
        for op_index in op_indices[len(values):]:
            set_tj(op_index, '')

    _write_content_stream(page, content)


def fill_resource_template(output_pdf_path: Path, panel_paths: list[Path], summary_stats: list[dict[str, float | None]] | None = None) -> None:
    boxes = [
        (10.8, 279.36, 242.04, 195.6),
        (240.36, 279.36, 242.04, 195.6),
        (469.8, 279.36, 242.04, 195.6),
        (699.24, 279.36, 242.04, 195.6),
        (19.44, 29.64, 242.04, 195.6),
        (248.88, 29.64, 242.04, 195.6),
        (478.32, 29.64, 242.04, 195.6),
        (707.76, 29.64, 242.16, 195.6),
    ]
    placements: list[tuple[Path | Image.Image, tuple[float, float, float, float]]] = list(zip(panel_paths, boxes))
    _replace_template_figures(
        TEMPLATE_ROOT / "AccuracyUnderResourceChange.pdf",
        output_pdf_path,
        placements,
        rewrite_hook=(lambda page, reader: _rewrite_resource_summary_text(page, reader, summary_stats)),
    )


def _expand_sampling_training_page(page, extra_bottom: float = 12.0) -> None:
    try:
        left = float(page.mediabox.left)
        bottom = float(page.mediabox.bottom)
        right = float(page.mediabox.right)
        top = float(page.mediabox.top)
        page.mediabox.lower_left = (left, bottom - extra_bottom)
        page.mediabox.upper_right = (right, top)
        if hasattr(page, "cropbox") and page.cropbox is not None:
            page.cropbox.lower_left = (left, bottom - extra_bottom)
            page.cropbox.upper_right = (right, top)
    except Exception:
        pass


def _expand_ours_overhead_page(page, extra_bottom: float = 26.0) -> None:
    try:
        left = float(page.mediabox.left)
        bottom = float(page.mediabox.bottom)
        right = float(page.mediabox.right)
        top = float(page.mediabox.top)
        page.mediabox.lower_left = (left, bottom - extra_bottom)
        page.mediabox.upper_right = (right, top)
        if hasattr(page, "cropbox") and page.cropbox is not None:
            page.cropbox.lower_left = (left, bottom - extra_bottom)
            page.cropbox.upper_right = (right, top)
    except Exception:
        pass


def _rewrite_ours_overhead_page(page, reader: PdfReader) -> None:
    _expand_ours_overhead_page(page)
    _remove_ours_overhead_labels(page, reader)


def _lower_sampling_training_captions(page, reader: PdfReader, delta_y: float = 10.0) -> None:
    content = ContentStream(page.get_contents(), reader)
    changed = False
    for idx, (operands, op) in enumerate(content.operations):
        if op != b'Tm' or len(operands) != 6:
            continue
        try:
            y_value = float(operands[5])
        except Exception:
            continue
        if 220.0 <= y_value <= 230.0:
            new_operands = list(operands)
            new_operands[5] = NumberObject(float(new_operands[5]) - delta_y)
            content.operations[idx] = (new_operands, op)
            changed = True
    if changed:
        _write_content_stream(page, content)


def fill_sampling_training_template(output_pdf_path: Path, panel_paths: list[Path]) -> None:
    boxes = [
        (4.32, 221.4, 237.12, 263.64),
        (238.92, 221.4, 237.24, 263.64),
        (480.0, 221.4, 237.24, 263.64),
        (721.08, 221.4, 237.24, 263.64),
    ]

    def rewrite(page, reader):
        _remove_sampling_training_bars(page, reader)
        _remove_sampling_training_text(page, reader)
        _expand_sampling_training_page(page, extra_bottom=12.0)
        _lower_sampling_training_captions(page, reader, delta_y=10.0)

    _replace_template_figures(
        TEMPLATE_ROOT / "SamplingTrainingBreakdown.pdf",
        output_pdf_path,
        zip(panel_paths, boxes),
        rewrite_hook=rewrite,
    )

def _remove_ours_overhead_labels(page, reader: PdfReader) -> None:
    content = ContentStream(page.get_contents(), reader)
    filtered = []
    changed = False
    last_tm = None
    index = 0
    ops = content.operations
    while index < len(ops):
        operands, op = ops[index]
        if op == b'Tm':
            last_tm = operands
            filtered.append((operands, op))
            index += 1
            continue
        if op in {b'Tj', b'TJ'} and last_tm is not None:
            try:
                y_value = float(last_tm[5])
            except Exception:
                y_value = None
            if y_value is not None and 300.0 <= y_value <= 410.0:
                changed = True
                index += 1
                continue
        if op == b're' and len(operands) == 4:
            try:
                x_value = float(operands[0])
                y_value = float(operands[1])
                w_value = float(operands[2])
                h_value = float(operands[3])
            except Exception:
                x_value = y_value = w_value = h_value = None
            if (
                x_value is not None
                and y_value is not None
                and w_value is not None
                and h_value is not None
                and abs(x_value) < 1.0
                and 300.0 <= y_value <= 410.0
                and 220.0 <= w_value <= 240.0
                and 20.0 <= h_value <= 26.0
            ):
                changed = True
                index += 1
                if index < len(ops) and ops[index][1] in {b'f', b'f*', b'B', b'B*', b'b', b'b*'}:
                    index += 1
                continue
        filtered.append((operands, op))
        index += 1
    if changed:
        content.operations = filtered
        stream = DecodedStreamObject()
        stream.set_data(content.get_data())
        page[NameObject('/Contents')] = stream


def fill_ours_overhead_template(output_pdf_path: Path, figure_image_path: Path) -> None:
    _replace_template_figures(
        TEMPLATE_ROOT / "OursOverheadBreakdown.pdf",
        output_pdf_path,
        [(figure_image_path, (80.88, 262.8, 427.08, 171.0))],
        rewrite_hook=_rewrite_ours_overhead_page,
    )


def fill_memory_template(output_pdf_path: Path, panel_paths: list[Path], summary_stats: list[dict[str, float | None]] | None = None) -> None:
    boxes = [
        (0.0, 273.72, 960.0, 200.04),
        (0.0, 55.92, 319.8, 199.8),
        (320.52, 55.92, 319.68, 199.8),
        (639.96, 55.92, 319.8, 199.8),
    ]
    if len(panel_paths) == 1:
        placements = [(panel_paths[0], _union(*boxes))]
    else:
        placements = list(zip(panel_paths, boxes))
    _replace_template_figures(
        TEMPLATE_ROOT / "Memory.pdf",
        output_pdf_path,
        placements,
        rewrite_hook=(lambda page, reader: (_remove_memory_annotations(page, reader), _rewrite_memory_summary_text(page, reader, summary_stats))),
    )


def fill_ablation_template(output_pdf_path: Path, figure_image_path: Path) -> None:
    _replace_template_figures(
        TEMPLATE_ROOT / "AbStudy.pdf",
        output_pdf_path,
        [(figure_image_path, (14.0, 144.0, 400.0, 378.0))],
        align='right',
        preserve_xobject_names={'/Image18', '/Image19'},
    )
