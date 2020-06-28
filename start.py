import ImageHandlerBot
import logging

if __name__ == "__main__":
    TOKEN = 'TOKEN FROM BOT FATHER'
    IM_SIZE = 256

    # Remove comments to Start logging
    #logging.basicConfig(level=logging.DEBUG,
    #                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    dialog_bot = ImageHandlerBot.ImageHandlerBot(TOKEN, IM_SIZE)
    dialog_bot.start()
