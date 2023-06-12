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


def send_message(message, chat_id):
    """
    Process and return image with new style
    """
    wid = str(uuid.uuid4())
    bot.send_message(chat_id, "Отправьте файл с основным изображением.")
    bot.register_next_step_handler(message, uploadfile_process, wid, 'main')
    bot.send_message(chat_id, "Отправьте изображение с стилем.")
    bot.register_next_step_handler(message, uploadfile_process, wid, 'style')
    bot.register_next_step_handler(message, files_process, wid)

    ################
def uploadfile_process(message, wid, type):
    bot.send_chat_action(message.chat.id, 'typing')
    try:
        os.makedirs(f'/tmp/{wid}', exist_ok=True)
        file_info = bot.get_file(message.photo[2].file_id)
        print(file_info.file_path)
        downloaded_file = bot.download_file(file_info.file_path)
        src = f"/tmp/{wid}/{type}_{file_info.file_path.split('/')[-1]}"
        print(file_info.file_path, src)
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.send_message(message.chat.id, "Файл успешно загружен")
    except Exception as e:
        print(str(e))
        bot.send_message(message.chat.id, "Ошибка! Отправьте файл как документ")
        #bot.register_next_step_handler(message, files_process)

def files_process(message, wid):
    print(f"TODO: Files processing {message.chat.id}: {wid}")

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
    bot.reply_to(message, f'Пойду, перенесу стиль с картинки на картинку, {message.from_user.first_name}')
    thread = threading.Thread(target=send_message, args=[message, message.chat.id])
    thread.start()

if __name__ == '__main__':
    # start_process()
    try:
        bot.polling(none_stop=True)
    except Exception: # pylint: disable=broad-except
        pass