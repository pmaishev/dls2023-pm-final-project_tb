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
#img.show()
digits = hashlib.md5(img.tobytes()).hexdigest()
print(digits)
assert('d474c601a5c6204c4a1f560500b3c10a'==digits)
links = {}
links['type'] = 'file'
links['content'] = './data_test/1/content_IMG_7359.jpg'
links['style'] = './data_test/1/style_IMG_7580.jpg'
img = trans.transfer(links)
#img.show()
digits = hashlib.md5(img.tobytes()).hexdigest()
print(digits)
assert('dceccefad2626f46e58b25c713c7b8bb'==digits)
