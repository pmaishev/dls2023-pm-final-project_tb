import sys
import hashlib
sys.path.insert(1, './')

from model.style_transfer import CStyleTransferConfig, CStyleTransfer

config = CStyleTransferConfig()
config.epoch = 5
config.max_size = 128

trans = CStyleTransfer(config)
links = {}
links['type'] = 'file'
links['content'] = './data_test/0/content_bruges.jpg'
links['style'] = './data_test/0/style_starry_night.jpg'
img = trans.transfer(links)
img.show()
digits = hashlib.md5(img.tobytes()).hexdigest()
print(digits)
assert('3bd537a6f59cd9bec6a0ab9b6bb546f7'==digits)

links['content'] = './data_test/1/content_IMG_7359.jpg'
links['style'] = './data_test/1/style_IMG_7580.jpg'
img = trans.transfer(links)
img.show()
digits = hashlib.md5(img.tobytes()).hexdigest()
print(digits)
#assert('82a632cb74a72afcc419d7f7eb910b4e'==digits)

links['content'] = './data_test/2/dancing.jpg'
links['style'] = './data_test/2/picasso.jpg'
img = trans.transfer(links)
img.show()
digits = hashlib.md5(img.tobytes()).hexdigest()
print(digits)
assert('cf0568e2b3e5b905e9723c8631d7c80c'==digits)
