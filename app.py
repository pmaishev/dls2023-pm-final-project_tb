#import time
import os
#from multiprocessing import *
#import schedule
import threading
import telebot

bot = telebot.TeleBot(os.environ["BOT_TOKEN"])

# def start_process():#Запуск Process
#     p1 = Process(target=P_schedule.start_schedule, args=()).start()


class CThread(): # Class для работы с schedule
    ####Функции для выполнения заданий по времени
    def send_message1(td, chat_id):
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
    bot.reply_to(message, f'Я бот. Приятно познакомиться, {message.from_user.first_name}')

@bot.message_handler(commands=['transfer_style'])
def send_welcome(message):
    cmd = message.text.split()
    if len(cmd) >= 2 and cmd[1].isdigit():
        diff = int(cmd[1])
    else:
        diff = 0
    bot.reply_to(message, f'Пойду, перенесу стиль с картинки на картинку, {message.from_user.first_name}')
    thread = threading.Thread(target=CThread.send_message1, args=(diff, message.chat.id))
    thread.start()

@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    if call.data == "cb_main":
        bot.answer_callback_query(call.id, "CB Main")
    elif call.data == "cb_style":
        bot.answer_callback_query(call.id, "CB Style")

if __name__ == '__main__':
    # start_process()
    try:
        bot.polling(none_stop=True)
    except:
        pass
