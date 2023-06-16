"""
Configuration file for bot
"""
import os

class CBotConfig():
    def __init__(self):
        self.token = os.environ["TG_BOT_TOKEN"]
        self.help_template = 'Я бот. Приятно познакомиться, {}'
        self.start_process_template = 'Пойду, перенесу стиль на изображение,  {}'
        self.content_upload_template = "Отправьте файл с основным изображением."
        self.style_upload_template = "Отправьте изображение с стилем."
    def format_file_url(self, file_info):
        return f'https://api.telegram.org/file/bot{self.token}/{file_info.file_path}'
