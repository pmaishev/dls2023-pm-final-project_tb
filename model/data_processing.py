"""
Module for data loading and saving
"""
import os

class IDataLoader():
    def load_data(self, wid: str, ftype: str):
        raise NotImplementedError()

class IDataSaver():
    def save_data(self, wid: str, ftype: str, name: str, data: bytes):
        raise NotImplementedError()

class CFileDataLoader(IDataLoader):
    def __init__(self, basedir: str='./tmp'):
        self.basedir = basedir

    def load_data(self, wid: str, ftype: str):
        for name in os.listdir(f'{self.basedir}/{wid}'):
            if name.startswith(ftype):
                break
        else:
            raise ValueError(f'{ftype} is not valid ftype. ftype must be content, style or mask')

        return open(f'{self.basedir}/{wid}/{name}', 'rb')

class CFileDataSaver(IDataSaver):
    def __init__(self, basedir: str='./tmp'):
        self.basedir = basedir

    def save_data(self, wid: str, ftype: str, name: str, data: bytes):
        os.makedirs(f'{self.basedir}/{wid}', exist_ok=True)
        src = f'{self.basedir}/{wid}/{ftype}_{name}'
        with open(src, 'wb') as file:
            file.write(data)
