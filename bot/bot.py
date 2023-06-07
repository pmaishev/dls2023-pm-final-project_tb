#import time
import telebot
from multiprocessing import *
#import schedule
import threading
import os

bot = telebot.TeleBot(os.environ["BOT_TOKEN"])

def start_process():#Запуск Process
    p1 = Process(target=P_schedule.start_schedule, args=()).start()
 
    
class P_schedule(): # Class для работы с schedule
    ####Функции для выполнения заданий по времени
    def send_message1(td, chat_id):
            bot.send_message(chat_id=chat_id, text='msg')
    ################

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, f'Я бот. Приятно познакомиться, {message.from_user.first_name}')

@bot.message_handler(commands=['test'])
def send_welcome(message):
    cmd = message.text.split()
    if len(cmd) >= 2 and cmd[1].isdigit():
        diff = int(cmd[1])
    else:
        diff = 0
    bot.reply_to(message, f'Пойду, подготовлю данные, {message.from_user.first_name}')
    x = threading.Thread(target=P_schedule.send_message1, args=(diff, message.chat.id))
    x.start()

if __name__ == '__main__':
    start_process()
    try:
        bot.polling(none_stop=True)
    except:
        pass
