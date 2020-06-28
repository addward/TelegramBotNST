from telegram.ext import Updater
from telegram.ext import MessageHandler, Filters, CallbackContext
from telegram import Document, PhotoSize, Message

import os
import concurrent.futures
import collections

import torch
import torchvision.models as models

import NST


class ImageHandlerBot(object):
    """
    Class for Telegram bot, implemented with Python Telegram Bot
    To create and start bot:
        dialog_bot = ImageHandlerBot.ImageHandlerBot(TOKEN, IMAGE_SIZE)
        dialog_bot.start()
    After that bot will handle messages using handle_message method and
    answer to user due to dialog generator function
    """

    def __init__(self, token, image_size):
        """
        Class constructor
        Parameters
        ----------
        token : unique token to control telegram bot
        image_size : size of output image
        """
        self.updater = Updater(token, use_context=True)
        self.job_queue = self.updater.job_queue

        self.handlers = collections.defaultdict(self.dialog)
        handler = MessageHandler(Filters.all, self.handle_message)
        self.updater.dispatcher.add_handler(handler)

		# For servers without cuda videocards
        device = torch.device("cpu")

        self.cnn = models.vgg19(pretrained=True).features.to(device).eval()

        self.cnn_normalization_mean = [0.485, 0.456, 0.406]
        self.cnn_normalization_std = [0.229, 0.224, 0.225]

        self.dict_running_transfers = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self.nst = NST.NeuralStyleTransfer(image_size, device, self.cnn,
                                           self.cnn_normalization_mean, self.cnn_normalization_std)

    def start(self):
        """
        Start telegram bot pooling
        """
        self.updater.start_polling()

    def dialog(self):
        """
        Generator function, that return the answer depend on the input message
        """

        yield from self._request_command('/start', 'Type /start to begin')
        yield from self._request_command('/transfer', "Hello, it's Neural Style Transfer bot for "
                                          "transferring style from one image to another "
                                          "with saved contest information of the second image. "
                                          "To start transfer type /transfer. To terminate our dialog "
                                          "at any stage type /quit")

        chat_id, image_content = yield from self._request_image('content')

        chat_id, image_style = yield from self._request_image('style')

        self.transfer_style(image_content, image_style, chat_id)

    @staticmethod
    def _load_image(image):
        """
        Method for image loading
        ----------
        image : telegram.PhotoSize or telegram.Document object that represent picture
        """
        return image.get_file().download()

    @staticmethod
    def _request_command(command_name, first_message):
        """
        Generator function that will permanently ask user to send command_name command
        Parameters
        ----------
        command_name : name of the disired command
        first_message: first message that will be send
        """

        answer = yield first_message
        while answer.text != '{}'.format(command_name):
            answer = yield "Type {} command".format(command_name)

    @staticmethod
    def _request_image(image_request_type):
        """
        Generator function that will permanently ask user to send Picture
        Parameters
        ----------
        image_request_type : image type that will be send in message to user
        """
        answer = yield "I'm waiting for {} image".format(image_request_type)
        attachment = answer.effective_attachment
        chat_id = answer.chat_id

        if type(attachment) == list:
            attachment = attachment[-1]

        while not (type(attachment) == PhotoSize or type(attachment) == Document):
            answer = yield "I don't understand your query, sent {} image one more time".format(image_request_type)
            # context.bot.sendMessage(chat_id=chat_id, text=answer)
            attachment = answer.effective_attachment
            if type(attachment) == list:
                attachment = attachment[-1]
        return chat_id, ImageHandlerBot._load_image(attachment)

    def handle_message(self, update, context):
        """
        Method that handles messages and send the response to user
        Parameters
        ----------
        update, context : parameters of user's message
        """
        chat_id = update.message.chat_id

        # In case of /quit command remove user's generator from handlers

        if update.message.text == "/quit":
            self.handlers.pop(chat_id)

        # If the user is not new send his message into generator

        if chat_id in self.handlers:
            try:
                answer = self.handlers[chat_id].send(update.message)
            except StopIteration:
                if chat_id in self.dict_running_transfers:
                    answer = 'Your picture is processing'
                else:
                    answer = self.restart_dialog(chat_id)

        # In other cases the user is new, so lets create handler for him

        else:
            next(self.handlers[chat_id])
            answer = self.handlers[chat_id].send(update.message)

        # And finally send generator response to user

        context.bot.sendMessage(chat_id=chat_id, text=answer)

    def restart_dialog(self, chat_id):
        """
        Restart dialog with user
        Parameters
        ----------
        chat_id : number of chat id with particular user
        """
        self.handlers.pop(chat_id)
        next(self.handlers[chat_id])
        return self.handlers[chat_id].send(Message(*[0] * 4, text='/start'))

    def transfer_style(self, pic_context, pic_style, chat_id):
        """
        Method to start transfer style
        Parameters
        ----------
        pic_context: path to contex picture
        pic_style  : path to style picture
        chat_id    : identification number of chat with user
        """

        # Add future object into queue of thread executor
        #self.nst.run_style_transfer(pic_context, pic_style)
        future = self.executor.submit(
            self.nst.run_style_transfer,
            pic_context,
            pic_style
        )

        # Insert future to dictionary of currently running futures and
        # add repeating job to check if this future finish style transfer

        self.dict_running_transfers[chat_id] = future
        #dict_final_pics[chat_id].exception(10)
        def check_image_ready(context, dict_final_pics, pictures_rm, chat_id):
            job = context.job
            if dict_final_pics[chat_id].done():
                output_picture = dict_final_pics[chat_id].result()
                dict_final_pics.pop(chat_id)
                context.bot.sendPhoto(chat_id=chat_id,
                                      photo=open(output_picture + '.jpg', 'rb'))
                context.bot.sendAnimation(chat_id=chat_id,
                                          animation=open(output_picture + '.gif', 'rb'))

                for pic in pictures_rm:
                    os.remove(pic)

                os.remove(output_picture + '.jpg')
                os.remove(output_picture + '.gif')

                job.schedule_removal()

        self.job_queue.run_repeating(
            lambda context: check_image_ready(context,
                                              self.dict_running_transfers,
                                              [pic_context, pic_style],
                                              chat_id), 5
        )
