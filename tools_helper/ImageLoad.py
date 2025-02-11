import base64
from io import BytesIO
from IPython.display import HTML, display
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
class ImageProcessor(object):
    def __init__(self):
        super().__init__()


    def image_load_path(self,path:str)->Image:
        '''
           通过传入路径进行图片文件的加载
        '''
        try:
            return Image.open(path)
        except IOError:
            return None

    def image_load_window(self)->Image:
        '''
          浏览本地窗口，打开文件夹下的图像,
          注意对应的返回值不是图像返回值为None
          '''
        try:
            root = tk.Tk()
            root.withdraw()
            file=filedialog.askopenfilename()
            if file is not None:
                if file.endswith(('jpg','png','jpeg','bmp')):
                    return Image.open(file)
                else:
                    return None
            else:
                return None
        except IOError:
            return None

    def convert_to_base64(self,pil_image:Image)->str:
        '''
          加载的图像转为base64
          错误码101：图片转成base64失败
          成功100：图片转换成功
          '''
        try:
            fromat=pil_image.format.lower()
            buffered = BytesIO()
            pil_image.save(buffered,format=fromat)  # You can change the format if needed
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str
        except Exception as e:
            return "101"+e

    def plt_img_base64(self,img_base64:str)->None:
        """
        可视化,pycharm中不能使用
        """
        # Create an HTML img tag with the base64 string as the source
        image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
        # Display the image by rendering the HTML
        display(HTML(image_html))

'''
测试代码

test = ImageProcessor()
image = test.image_load_window()
str1=test.convert_to_base64(image)
test.plt_img_base64(str1)
'''
