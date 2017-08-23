import os
import time
import numpy as np

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import StringProperty, NumericProperty, ReferenceListProperty
from kivy.vector import  Vector
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image

from PIL import Image

#Window.clearcolor = (1, 1, 1, 1)

# モデルの読み込み
model = VGG16(weights='imagenet')
# 入力画像のロード
def predict(filename):
    img = image.load_img(filename, target_size=(224, 224))
    # 入力画像の行列化
    x = image.img_to_array(img)
    # 4次元テンソル
    x = np.expand_dims(x, axis=0)
    # 予測
    preds = model.predict(preprocess_input(x))
    results = decode_predictions(preds, top=3)[0]
    # 結果出力
    os.remove(filename)
    return results
    #return results[0][1], str(round(results[0][2] * 100,2))


class CameraClick(Widget):
    name1 = StringProperty()
    prob1 = StringProperty()
    name2 = StringProperty()
    prob2 = StringProperty()
    name3 = StringProperty()
    prob3 = StringProperty()
    x1 = NumericProperty(0)
    y1 = NumericProperty(0)
    x2 = NumericProperty(0)
    y2 = NumericProperty(0)
    x3 = NumericProperty(0)
    y3 = NumericProperty(0)
    bar1 = ReferenceListProperty(x1, y1)
    bar2 = ReferenceListProperty(x2, y2)
    bar3 = ReferenceListProperty(x3, y3)
    source = StringProperty('./images/noimage.png')
    def capture(self):
        camera = self.ids['camera']
        prob1obj = self.ids['prob1']
        prob2obj = self.ids['prob2']
        prob3obj = self.ids['prob3']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        filename = "./tmp/IMG_{}.png".format(timestr)
        camera.export_to_png(filename)
        img = Image.open(filename)

        # 切り抜き(left, upper right, lower)
        img.crop((40, 90, 460 ,370)).save(filename)
        self.source = filename
        results = predict(filename)
        prob1 = round(results[0][2], 2)
        prob2 = round(results[1][2], 2)
        prob3 = round(results[2][2], 2)
        self.name1, self.prob1=results[0][1], str(int(prob1 * 100)) + "%"
        self.name2, self.prob2 = results[1][1], str(int(prob2 * 100)) + "%"
        self.name3, self.prob3=results[2][1], str(int(prob3 * 100)) + "%"
        self.bar1 = [int(prob1obj.size[0] * prob1), 10]
        self.bar2 = [int(prob2obj.size[0] * prob2), 10]
        self.bar3 = [int(prob3obj.size[0] * prob3),10]
        #self.name , self.prob = predict(filename)
        print("Captured")

    def clear_image(self):
        self.source = './images/noimage.png'
        self.name1, self.prob1="", ""
        self.name2, self.prob2 = "", ""
        self.name3, self.prob3="", ""
        self.bar1 = [0,0]
        self.bar2 = [0,0]
        self.bar3 = [0,0]

class Camera(App):

    def build(self):
        return CameraClick()


if __name__ == '__main__':
    Camera().run()
