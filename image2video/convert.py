from PIL import Image


def resize_image(input_image_path, output_image_path, size=(768, 768)):
    with Image.open(input_image_path) as img:
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(output_image_path)


input_path = "/dom/image2video/20240119_12_38_0.png"
output_path = "/dom/image2video/20240119_12_38_0_convert.png"

resize_image(input_path, output_path)
