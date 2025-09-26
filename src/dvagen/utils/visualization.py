from PIL import Image, ImageDraw, ImageFont


def get_gradient_color(
    base_rgb: tuple[int, int, int], prob: float
) -> tuple[int, int, int]:
    light_rgb = (
        int(base_rgb[0] * 0.4 + 255 * 0.6),
        int(base_rgb[1] * 0.4 + 255 * 0.6),
        int(base_rgb[2] * 0.4 + 255 * 0.6),
    )

    r = int(light_rgb[0] * (1 - prob) + base_rgb[0] * prob)
    g = int(light_rgb[1] * (1 - prob) + base_rgb[1] * prob)
    b = int(light_rgb[2] * (1 - prob) + base_rgb[2] * prob)

    return (r, g, b)


def get_visualization(
    output: list[dict],
    output_image_path: str | None = None,
    scaling_level: float = 1.0,
    show_text: bool = False,
    **kwargs,
) -> Image.Image:
    # Phrase Input
    tokens_or_phrases = [item["token"] for item in output]
    types = [item["type"] == "phrase" for item in output]
    probs = [item["prob"] for item in output]

    # region Arguments
    # Picture Constrants
    base_image_width: int = 600
    base_font_size: int = 18
    base_line_height: int = 20
    base_padding_x: int = 4
    base_padding_y: int = 4
    base_border_padding: int = 10
    base_colorbar_height: int = 20
    base_colorbar_label_size: int = 18
    base_colorbar_label_padding: int = 5
    base_colorbar_margin_top: int = 0
    base_colorbar_padding: int = 5
    base_colorbar_margin_bottom: int = 8

    # Scale Level
    image_width = int(base_image_width * scaling_level)
    line_height = int(base_line_height * scaling_level)
    content_font_size = int(base_font_size * scaling_level)
    padding_x = int(base_padding_x * scaling_level)
    padding_y = int(base_padding_y * scaling_level)
    border_padding = int(base_border_padding * scaling_level)
    colorbar_height = int(base_colorbar_height * scaling_level)
    colorbar_label_size = int(base_colorbar_label_size * scaling_level)
    colorbar_label_padding = int(base_colorbar_label_padding * scaling_level)
    colorbar_margin_top = int(base_colorbar_margin_top * scaling_level)
    colorbar_padding = int(base_colorbar_padding * scaling_level)
    colorbar_margin_bottom = int(base_colorbar_margin_bottom * scaling_level)

    # Block Colors
    token_color_dark = kwargs.pop("token_color_dark", (255, 232, 20))
    phrase_color_dark = kwargs.pop("phrase_color_dark", (0, 204, 204))

    # font style
    content_font_family = kwargs.pop("content_font_family", None)
    content_font_color = kwargs.pop("content_font_color", (0, 0, 0))
    label_font_family = kwargs.pop("label_font_family", None)
    label_font_color = kwargs.pop("label_font_color", (255, 255, 255))
    # endregion

    # region Pre-Calc
    # font
    if label_font_family:
        label_font = ImageFont.truetype(
            font=label_font_family,
            size=colorbar_label_size,
        )
    else:
        label_font = ImageFont.load_default(colorbar_label_size)
    if content_font_family:
        content_font = ImageFont.truetype(
            font=content_font_family,
            size=content_font_size,
        )
    else:
        content_font = ImageFont.load_default(content_font_size)

    current_x = 0
    lines = []  # multiple lines of blocks
    current_line_elements = []

    for i, token in enumerate(tokens_or_phrases):
        # get font
        text_bbox = content_font.getbbox(token)
        text_width = text_bbox[2]
        text_height = line_height
        base_color = phrase_color_dark if types[i] else token_color_dark

        # block element
        element = {
            "text": token,
            "prob": probs[i],
            "type": types[i],
            "width": text_width,
            "height": text_height,
            "base_color": base_color,
        }

        # check for new line
        if (
            current_x + element["width"] + padding_x > image_width
            and current_x != 0
        ):
            lines.append(current_line_elements)
            current_line_elements = []
            current_x = 0

        # append block into current line
        current_line_elements.append(element)
        current_x += element["width"] + padding_x

    # append the last line
    if current_line_elements:
        lines.append(current_line_elements)

    # Calculate Total Content Height
    total_content_height = (
        (len(lines) * (line_height + padding_y) - padding_y) if lines else 0
    )

    # Pictue Final Size
    colorbar_total_height = (
        colorbar_margin_top
        + colorbar_height
        + colorbar_padding
        + colorbar_height
        + colorbar_margin_bottom
    )
    final_image_width = border_padding + image_width + border_padding
    final_image_height = (
        border_padding
        + total_content_height
        + border_padding
        + colorbar_total_height
    )

    # endregion

    # region Draw Picture
    image = Image.new("RGB", (final_image_width, final_image_height), "white")
    draw = ImageDraw.Draw(image)

    # region Color Bar Region
    colorbar_start_y = border_padding + colorbar_margin_top

    # region Token Color Bar
    token_start_y = colorbar_start_y
    # Bar
    for x_idx in range(image_width):
        prob = x_idx / image_width
        color = get_gradient_color(token_color_dark, prob)
        draw.line(
            [
                (border_padding + x_idx, token_start_y),
                (
                    border_padding + x_idx,
                    token_start_y + colorbar_height - 1,
                ),
            ],
            fill=color,
            width=1,
        )
    # Label
    draw.text(
        xy=(
            border_padding
            + image_width / 2
            - label_font.getbbox("Token Probability")[2] / 2,
            token_start_y
            + colorbar_height / 2
            - label_font.getbbox("Token Probability")[3] / 2,
        ),
        text="Token Probability",
        font=label_font,
        fill=label_font_color,
    )
    # 0.0
    draw.text(
        xy=(
            border_padding + colorbar_label_padding,
            token_start_y
            + colorbar_height / 2
            - label_font.getbbox("0.0")[3] / 2,
        ),
        text="0.0",
        font=label_font,
        fill=label_font_color,
    )
    # 1.0
    draw.text(
        xy=(
            final_image_width
            - border_padding
            - label_font.getbbox("1.0")[2]
            - colorbar_label_padding,
            token_start_y
            + colorbar_height / 2
            - label_font.getbbox("1.0")[3] / 2,
        ),
        text="1.0",
        font=label_font,
        fill=label_font_color,
    )
    # endregion

    # region Prase Color Bar
    phrase_start_y = token_start_y + colorbar_height + colorbar_padding

    # Bar
    for x_idx in range(image_width):
        prob = x_idx / image_width
        color = get_gradient_color(phrase_color_dark, prob)
        draw.line(
            [
                (border_padding + x_idx, phrase_start_y),
                (
                    border_padding + x_idx,
                    phrase_start_y + colorbar_height - 1,
                ),
            ],
            fill=color,
            width=1,
        )
    # Label
    draw.text(
        xy=(
            border_padding
            + image_width / 2
            - label_font.getbbox("Phrase Probability")[2] / 2,
            phrase_start_y
            + colorbar_height / 2
            - label_font.getbbox("Phrase Probability")[3] / 2,
        ),
        text="Phrase Probability",
        font=label_font,
        fill=label_font_color,
    )
    # 0.0
    draw.text(
        xy=(
            border_padding + colorbar_label_padding,
            phrase_start_y
            + colorbar_height / 2
            - label_font.getbbox("0.0")[3] / 2,
        ),
        text="0.0",
        font=label_font,
        fill=label_font_color,
    )
    # 1.0
    draw.text(
        xy=(
            final_image_width
            - border_padding
            - label_font.getbbox("1.0")[2]
            - colorbar_label_padding,
            phrase_start_y
            + colorbar_height / 2
            - label_font.getbbox("1.0")[3] / 2,
        ),
        text="1.0",
        font=label_font,
        fill=label_font_color,
    )
    # endregion

    # endregion

    content_start_y = (
        colorbar_start_y
        + colorbar_height
        + colorbar_padding
        + colorbar_height
        + colorbar_margin_bottom
    )
    for line in lines:
        current_x_draw = border_padding
        for element in line:
            # get gradient color
            fill_color = get_gradient_color(
                element["base_color"], element["prob"]
            )

            # draw block
            draw.rectangle(
                xy=[
                    current_x_draw,
                    content_start_y,
                    current_x_draw + element["width"],
                    content_start_y + element["height"],
                ],
                fill=fill_color,
            )

            # show text
            if show_text:
                text_x = (
                    current_x_draw
                    + (
                        element["width"]
                        - content_font.getbbox(element["text"])[2]
                    )
                    / 2
                )
                text_y = (
                    content_start_y
                    + (
                        element["height"]
                        - content_font.getbbox(element["text"])[3]
                    )
                    / 2
                )
                draw.text(
                    xy=(text_x, text_y),
                    text=element["text"],
                    font=content_font,
                    fill=(120, 120, 120)
                    if element["prob"] < 0.7
                    else (255, 255, 255),
                )

            current_x_draw += element["width"] + padding_x

        content_start_y += line_height + padding_y
    # endregion

    # Save Picture
    if output_image_path:
        image.save(output_image_path)
        print(f"Picture Saving to {output_image_path}")

    # Show
    return image
