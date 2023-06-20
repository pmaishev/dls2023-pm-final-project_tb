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


def style_transfer_message(message, content_url, style_url):
    """
    Process and return image with new style
    """
    links = {}
    links['type'] = 'url'
    links['content'] = content_url
    links['style'] = style_url

    print(f"Start files processing {message.chat.id}: {content_url.replace(config.token, 'XXXXX')} {style_url.replace(config.token, 'XXXXX')}")
    style_transfer = CStyleTransfer()

    bot.send_photo(message.chat.id, photo=style_transfer.transfer(links))
    bot.send_message(message.chat.id, "Done")

    ################
def upload_main_file_process(message):
    bot.send_chat_action(message.chat.id, 'typing')
    try:
        file_info = bot.get_file(message.photo[2].file_id)
        content_url = config.format_file_url(file_info)
        message = bot.reply_to(message, config.style_upload_template)
        bot.register_next_step_handler(message, upload_style_file_process, content_url)
    except Exception as e:
        print(str(e))
        bot.send_message(message.chat.id, "Ошибка! Отправьте основное изображение.")
        bot.register_next_step_handler(message, upload_main_file_process)

def upload_style_file_process(message, content_url):
    bot.send_chat_action(message.chat.id, 'typing')
    try:
        file_info = bot.get_file(message.photo[2].file_id)
        style_url = config.format_file_url(file_info)
        bot.reply_to(message, config.start_process_template.format(message.from_user.first_name))
        thread = threading.Thread(target=style_transfer_message, args=[message, content_url, style_url])
        thread.start()
    except Exception as e:
        print(str(e))
        bot.send_message(message.chat.id, "Ошибка! Отправьте изображение стиля.")
        bot.register_next_step_handler(message, upload_style_file_process, content_url)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """
    Welcome message from bot.
    """
    bot.reply_to(message, config.help_template.format(message.from_user.first_name))
    if config.transfer_config.device.type == 'cpu':
        bot.send_message(message.chat.id, config.sorry_for_cpu_template)

@bot.message_handler(commands=['transfer_style'])
def send_transfered_image(message):
    """
    Upload image and process it
    """
    message = bot.reply_to(message, config.content_upload_template)
    bot.register_next_step_handler(message, upload_main_file_process)

if __name__ == '__main__':
    # start_process()
    try:
        bot.polling(none_stop=True)
    except Exception: # pylint: disable=broad-except
        pass