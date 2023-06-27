'''
Configuration file for bot
'''
import os
from model.style_transfer import CStyleTransferConfig

class CBotConfig():
    def __init__(self):
        self.token = os.environ['TG_BOT_TOKEN']
        self.help_template = '''Я бот, написанный в рамках итогового проекта https://stepik.org/course/135003. Приятно познакомиться, {}.
Я умею переносить стиль с одного изображения на другое.
Реализованно на основе https://arxiv.org/abs/1703.06953 и https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer.
Для начала работы надо ввести /transfer_style, а потом загрузить основное изображение и изображение с стилем.

Также я умею переносить стиль "медленным" способом. Реализованно по алгоритму, описанному в https://arxiv.org/abs/1508.06576 и его реализации в https://pytorch.org/tutorials/advanced/neural_style_tutorial.html.
Для медленного переноса стиля надо ввести /transfer_style_slow.
'''
        self.sorry_for_cpu_template = '''К сожалению, я работаю на VPS без GPU, поэтому:
1) размер выходного изображения для медленного способа будет 128 px. Время ожидания ~ 1 час
2) размер выходного изображения для быстрого способа будет 1024 px. Время ожидания ~ 2 минуты
'''
        self.start_process_template = 'Пойду, перенесу стиль на изображение, {}'
        self.content_upload_template = 'Отправьте файл с основным изображением.'
        self.style_upload_template = 'Отправьте изображение с стилем.'
        self.content_error_template =  'Ошибка! Отправьте основное изображение.'
        self.style_error_template = 'Ошибка! Отправьте изображение стиля.'
        self.transfer_config = CStyleTransferConfig()
    def format_file_url(self, file_info):
        return f'https://api.telegram.org/file/bot{self.token}/{file_info.file_path}'
