import os
from PIL import Image

# 定义图片文件夹路径
folder_path = ("your_fold+"
               ""
               ""
               ""
               ""
               ""
               "3.er_path")  # 替换为你的文件夹路径

# 获取文件夹中所有文件名
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]  # 只选择图片文件

# 循环遍历文件夹中的每张图片
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)  # 获取完整路径
    img = Image.open(image_path)  # 加载图片
    img.show()  # 显示图片（如果需要）

    # 可以在这里对图片做其他操作，例如处理、分析等

    # 你可以加一个延时，或者通过一些条件来控制如何继续加载下一张图片
    # 例如，加入输入等待，手动确认加载下一张图片
    input("按 Enter 键加载下一张图片...")  # 按回车继续加载下一张图片