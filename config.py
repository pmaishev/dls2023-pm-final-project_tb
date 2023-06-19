"""
Configuration file for bot
"""
import os
from model.style_transfer import CStyleTransferConfig

class CBotConfig():
    def __init__(self):
        self.token = os.environ["TG_BOT_TOKEN"]
        self.help_template = '''Я бот. Приятно познакомиться, {}.
        Я умею переносить стиль с одного изображения на другое.
        Сделан по алгоритму, описанному в https://arxiv.org/abs/1508.06576 и его реализации в https://pytorch.org/tutorials/advanced/neural_style_tutorial.html.
        Для начала работы надо ввести /transfer_style, а потом загрузить основное изображение и изображение с стилем.'''
        self.sorry_for_cpu_template = 'К сожалению, я работаю на VPS без GPU, поэтому размер выходного изображения будет 128 px. Время ожидания ~ 40 минут'
        self.start_process_template = 'Пойду, перенесу стиль на изображение,  {}'
        self.content_upload_template = "Отправьте файл с основным изображением."
        self.style_upload_template = "Отправьте изображение с стилем."
        self.transfer_config = CStyleTransferConfig()
    def format_file_url(self, file_info):
        return f'https://api.telegram.org/file/bot{self.token}/{file_info.file_path}'
