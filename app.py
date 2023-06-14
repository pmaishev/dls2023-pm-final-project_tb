"""
Telegram bot for style transfering
"""
import time
import os
#from multiprocessing import *
#import schedule
import threading
import uuid
import telebot
from config import CBotConfig
from model.style_transfer import CStyleTransfer

config = CBotConfig()
bot = telebot.TeleBot(config.token)

# def start_process():#Запуск Process
#     p1 = Process(target=P_schedule.start_schedule, args=()).start()


def style_transfer_message(message, wid):
    """
    Process and return image with new style
    """
    print(f"TODO: Files processing {message.chat.id}: {wid}")
    style_transfer = CStyleTransfer()

    bot.send_message(message.chat.id, "Start")
    bot.send_photo(message.chat.id, photo=style_transfer.transfer(wid))
    bot.send_message(message.chat.id, "Done")

    ################
def upload_main_file_process(message, wid):
    bot.send_chat_action(message.chat.id, 'typing')
    try:
        upload_file(message, wid, 'content')
        message = bot.reply_to(message, "Отправьте изображение с стилем.")
        bot.register_next_step_handler(message, upload_style_file_process, wid)
    except Exception as e:
        print(str(e))
        bot.send_message(message.chat.id, "Ошибка! Отправьте изображение")
        #bot.register_next_step_handler(message, files_process)

def upload_style_file_process(message, wid):
    bot.send_chat_action(message.chat.id, 'typing')
    try:
        upload_file(message, wid, 'style')
        bot.reply_to(message, f'Пойду, перенесу стиль с картинки на картинку, {message.from_user.first_name}')
        thread = threading.Thread(target=style_transfer_message, args=[message, wid])
        thread.start()
    except Exception as e:
        print(str(e))
        bot.send_message(message.chat.id, "Ошибка! Отправьте изображение стиля.")
        #bot.register_next_step_handler(message, files_process)

def upload_file(message, wid, ftype):
    file_info = bot.get_file(message.photo[2].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    config.get_data_saver().save_data(wid, ftype, file_info.file_path.split('/')[-1], downloaded_file)
    bot.send_message(message.chat.id, "Файл успешно загружен")

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """
    Welcome message from bot.
    """
    bot.reply_to(message, f'Я бот. Приятно познакомиться, {message.from_user.first_name}')

@bot.message_handler(commands=['transfer_style'])
def send_transfered_image(message):
    """
    Upload image and process it
    """
    wid = str(uuid.uuid4())
    message = bot.reply_to(message, "Отправьте файл с основным изображением.")
    bot.register_next_step_handler(message, upload_main_file_process, wid)

if __name__ == '__main__':
    # start_process()
    try:
        bot.polling(none_stop=True)
    except Exception: # pylint: disable=broad-except
        pass