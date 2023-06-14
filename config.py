"""
Configuration file for bot
"""
import os
from model.data_processing import IDataSaver, CFileDataSaver

class CBotConfig():
    def __init__(self):
        self.data_saver = CFileDataSaver()
        self.token = os.environ["TG_BOT_TOKEN"]

    def get_data_saver(self) -> IDataSaver:
        return self.data_saver
