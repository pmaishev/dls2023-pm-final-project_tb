"""
Telegram bot for style transfering
"""
#import time
import os
#from multiprocessing import *
#import schedule
import threading
import uuid
import telebot

bot = telebot.TeleBot(os.environ["TG_BOT_TOKEN"])

# def start_process():#Запуск Process
#     p1 = Process(target=P_schedule.start_schedule, args=()).start()


def send_message(chat_id):
    """
    Process and return image with new style
    """
    bot.send_message(chat_id=chat_id, text='msg')
    button_foo = telebot.types.InlineKeyboardButton('Main image', callback_data='cb_main')
    button_bar = telebot.types.InlineKeyboardButton('Style image', callback_data='cb_style')

    keyboard = telebot.types.InlineKeyboardMarkup()
    keyboard.add(button_foo)
    keyboard.add(button_bar)

    bot.send_message(chat_id, text='Keyboard example', reply_markup=keyboard)
    ################

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
#    cmd = message.text.split()
    bot.reply_to(message, f'Пойду, перенесу стиль с картинки на картинку, {message.from_user.first_name}')
    thread = threading.Thread(target=send_message, args=[message.chat.id])
    thread.start()

@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    """
    Buttons callbacks
    """
    if call.data == "cb_main":
        @bot.message_handler(content_types=['photo'])
        def photo_processing(message):
            file_info = bot.get_file(message.photo.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            with open(f'tmp/{uuid.uuid4()}.jpg', 'wb') as new_file:
                new_file.write(downloaded_file)
            #bot.answer_callback_query(call.id, "CB Main")
    elif call.data == "cb_style":
        bot.answer_callback_query(call.id, "CB Style")

if __name__ == '__main__':
    # start_process()
    try:
        bot.polling(none_stop=True)
    except Exception: # pylint: disable=broad-except
        pass
