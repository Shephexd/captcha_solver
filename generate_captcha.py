from io import BytesIO
from captcha.image import ImageCaptcha
import random
import sys
font_set = ['font/captcha0.ttf', 'font/captcha1.ttf', 'font/captcha2.ttf', 'font/captcha3.ttf', 'font/captcha4.ttf', 'font/captcha5.ttf']
image = ImageCaptcha(fonts=font_set)
#number = '12345'

if len(sys.argv) >= 2:
    N = int(sys.argv[1])
else:
    N = 1

for i in range(N):
    number = str(random.randrange(10000,99999))
    data = image.generate(number)
    assert isinstance(data, BytesIO)
    image.write(number, 'data/{}.png'.format(number))
    print("generate {}".format(number))

print("generate {} samples".format(N))
