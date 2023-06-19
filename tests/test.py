import sys
import hashlib
sys.path.insert(1, './')

from model.style_transfer import CStyleTransferConfig, CStyleTransfer

config = CStyleTransferConfig()
config.epoch = 0
config.max_size = 128

trans = CStyleTransfer(config)
links = {}
links['type'] = 'file'
links['content'] = './data_test/0/content_bruges.jpg'
links['style'] = './data_test/0/style_starry_night.jpg'
img = trans.transfer(links)
#img.show()
digits = hashlib.md5(img.tobytes()).hexdigest()
print(digits)
assert('8a382336c4ef879fbe1f32768d8dc867'==digits)

links['content'] = './data_test/1/content_IMG_7359.jpg'
links['style'] = './data_test/1/style_IMG_7580.jpg'
img = trans.transfer(links)
#img.show()
digits = hashlib.md5(img.tobytes()).hexdigest()
print(digits)
assert('81496fac33a903d86f3274ac02988c5a'==digits)

links['content'] = './data_test/2/dancing.jpg'
links['style'] = './data_test/2/picasso.jpg'
img = trans.transfer(links)
#img.show()
digits = hashlib.md5(img.tobytes()).hexdigest()
print(digits)
assert('5408f69c00fe468a7a76b133b834f298'==digits)
