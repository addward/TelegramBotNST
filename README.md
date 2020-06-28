# TelegramBotNST
Python telegram bot for image style transfer using vgg19
<br/>Style transfer implementation is based on 
      [pytorch tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) and 
      [article](https://arxiv.org/abs/1508.06576)
<br/>Telegram bot is implemented using [Python Telegram Bot library](https://github.com/python-telegram-bot/python-telegram-bot)
## How to start
1) Recieve token from [Bot Father](https://t.me/botfather)
2) Type recieved token and desired output image size into start.py file
3) Start start.py script (sudo python3 start.py)
## Examples
|Content image|Style image|Output|
|---|---|---|
|![](https://github.com/addward/NST_2pics/blob/master/Examples/oak_in.jpg)|![](https://github.com/addward/NST_2pics/blob/master/Examples/munk.jpg)|![](https://github.com/addward/NST_2pics/blob/master/Examples/oak_munk.gif)|
|![](https://github.com/addward/NST_2pics/blob/master/Examples/oak_in.jpg)|![](https://github.com/addward/NST_2pics/blob/master/Examples/mozaic.jpg)|![](https://github.com/addward/NST_2pics/blob/master/Examples/oak_out.gif)|
|![](https://github.com/addward/NST_2pics/blob/master/Examples/van_gogh_in.jpg)|![](https://github.com/addward/NST_2pics/blob/master/Examples/mozaic.jpg)|![](https://github.com/addward/NST_2pics/blob/master/Examples/van_gogh_out.gif)|