import sys
import hashlib
sys.path.insert(1, './')

from model.data_processing import CFileDataLoader
from model.style_transfer import CStyleTransferConfig, CStyleTransfer

config = CStyleTransferConfig()
config.data_loader = CFileDataLoader(basedir='./data_test')
config.epoch = 5
config.max_size = 128

trans = CStyleTransfer(config)

img = trans.transfer('11111111-1111-1111-1111-11111111111')
#img.show()
digits = hashlib.md5(img.tobytes()).hexdigest()
print(digits)
assert('dceccefad2626f46e58b25c713c7b8bb'==digits)

img = trans.transfer('00000000-0000-0000-0000-000000000000')
#img.show()
digits = hashlib.md5(img.tobytes()).hexdigest()
print(digits)
assert('d474c601a5c6204c4a1f560500b3c10a'==digits)
