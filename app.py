"""
Telegram bot for style transfering
"""
import threading
import telebot
from config import CBotConfig
from model.style_transfer import CStyleTransfer
from model.style_transfer_msg import CStyleTransferMsg

config = CBotConfig()
bot = telebot.TeleBot(config.token)

# def start_process():#Запуск Process
#     p1 = Process(target=P_schedule.start_schedule, args=()).start()


def style_transfer_message(message, content_url, style_url, target):
    """
    Process and return image with new style
    """
    links = {}
    links['type'] = 'url'
    links['content'] = content_url
    links['style'] = style_url

    print(f"Start files processing {type(target)} {message.chat.id}: {content_url.replace(config.token, 'XXXXX')} {style_url.replace(config.token, 'XXXXX')}")
    style_transfer = target()

    bot.send_photo(message.chat.id, photo=style_transfer.transfer(links))

    ################
def upload_main_file_process(message, target):
    """
    Upload image with content
    """
    bot.send_chat_action(message.chat.id, 'typing')
    try:
        file_info = bot.get_file(message.photo[2].file_id)
        content_url = config.format_file_url(file_info)
        message = bot.reply_to(message, config.style_upload_template)
        bot.register_next_step_handler(message, upload_style_file_process, content_url, target)
    except Exception as e:
        print(str(e))
        bot.send_message(message.chat.id, config.content_error_template)
        bot.register_next_step_handler(message, upload_main_file_process, target)

def upload_style_file_process(message, content_url, target):
    """
    Upload image with style
    """
    bot.send_chat_action(message.chat.id, 'typing')
    try:
        file_info = bot.get_file(message.photo[2].file_id)
        style_url = config.format_file_url(file_info)
        bot.reply_to(message, config.start_process_template.format(message.from_user.first_name))
        thread = threading.Thread(target=style_transfer_message, args=[message, content_url, style_url, target])
        thread.start()
    except Exception as e:
        print(str(e))
        bot.send_message(message.chat.id, config.style_upload_template)
        bot.register_next_step_handler(message, upload_style_file_process, content_url, target)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """
    Welcome message from bot.
    """
    bot.reply_to(message, config.help_template.format(message.from_user.first_name))
    if config.transfer_config.device.type == 'cpu':
        bot.send_message(message.chat.id, config.sorry_for_cpu_template)

@bot.message_handler(commands=['transfer_style', 'transfer_style_slow'])
def send_transfered_image(message):
    """
    Upload image and process it
    """
    target = CStyleTransferMsg if message.text.split()[0] == '/transfer_style' else CStyleTransfer
    message = bot.reply_to(message, config.content_upload_template)
    bot.register_next_step_handler(message, upload_main_file_process, target)

if __name__ == '__main__':
    # start_process()
    try:
        bot.polling(none_stop=True)
    except Exception: # pylint: disable=broad-except
        pass