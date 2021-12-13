# -*- coding: utf-8 -*-
from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QSplitter,\
    QComboBox, QLabel, QFrame, QApplication, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap
from PaintBoard import PaintBoard
import random



import PIL
import torch
import torchvision
from torchvision import transforms, utils
from torch.autograd import Variable


#預測結果的部分
def conv3x3(in_channels, out_channels, stride = 1):
    
    return torch.nn.Conv2d(in_channels, out_channels,
                           kernel_size = 3, stride = stride,
                           padding = 1, bias = False)


#要讀取 model 出來必須得先建一個一樣的
class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        
        super(ResidualBlock, self).__init__()
        
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        
        
    def forward(self, x):
        
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out = residual + out
        out = self.relu(out)
        
        
        return out



class ResNet(torch.nn.Module):
    
    def __init__(self, block, layers, num_classes = 102):
        
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        self.conv = conv3x3(3, 64)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace = True)
        
        
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)
        self.layer4 = self.make_layer(block, 512, layers[3], 2)

        
        self.avg_pool = torch.nn.AvgPool2d(4)
        
        self.fc = torch.nn.Linear(512, num_classes)
        
        

    def make_layer(self, block, out_channels, blocks, stride = 1):

        downsample = None
        
        if((stride != 1) or (self.in_channels != out_channels)):
            
            downsample = torch.nn.Sequential(
                    conv3x3(self.in_channels, out_channels, stride),
                    torch.nn.BatchNorm2d(out_channels)
                )
        
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        self.in_channels = out_channels
        
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return torch.nn.Sequential(*layers)


    def forward(self, x):
        
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        
        
        out = self.avg_pool(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out




class MainWidget(QWidget):   

    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)
        
        
        self.setFixedSize(900,600)
        self.setWindowTitle("Dr. Draw")
        
        self.appeared = []
        self.total_question = 1
        
        self.topics = ["airplane", "ambulance","apple", "axe", "banana", "baseball bat",
                "basketball", "bathtub", "bed", "bicycle", "book", "bridge", "brocolli",
                "bucket", "butterfly", "cake", "camel", "campfire", "circle",
                "clock", "cloud", "computer", "cookie", "crab", "crocodile", "cup",
                "diamond", "dolphin", "donut",  "door", "drums", "ear", "envelope", "eye",
                "eyeglasses", "feather", "finger", "fish", "flower", "fork", "giraffe",
                "grass", "guitar", "hexagon", "hospital", "house", "ice cream",
                "jail", "knife","leaf", 'light bulb', "lightning", 'line', 'lion', 'lollipop', 
                'mermaid', 'moon', 'mosquito', 'ocean', "octopus", 'onion', 'paintbrush', 
                'panda', 'pencil', 'penguin', 'piano', "pig", 'pineapple', 'pizza', 
                'potato', 'rain', 'rainbow', 'river', 'sandwich', "saxophone", 'scissors', 
                "sea turtle", "see saw", 'shark', "sheep", 'shoe', 'snake', 'snowman', "square",
                'stairs', 'star', 'strawberry', 'sun', 'sword', 'television', 'tent', 'tiger', 'toothbrush', 
                'train', 'tree', "triangle", 'umbrella', 'violin', 'watermelon', 'whale', "windmill", " "]
        
        #print(len(self.topics))
        self.score = 0        
        
        self.i = 0; #目前題目
        self.game_timer = QTimer(self) #遊戲計時器
        self.remain_timer = QTimer(self) #遊戲剩下的時間
        self.__start()

        #print(self.topics)
        
    def __InitData(self):
        '''
        初始化成员变量
        '''
        self.__paintBoard = PaintBoard(self)
        #获取颜色列表(字符串类型)
        self.__colorList = QColor.colorNames()
        self.__InitView()
    
    #開始畫面
    def __start(self):
        self.frame = QFrame(self)
        self.frame.resize(900, 600)
        start_button = QPushButton("開始遊戲")
        start_button.setParent(self.frame)
        start_button.setGeometry(400,500,80,30)
        start_button.clicked.connect(self.__InitData)        
        
        pic = QLabel(self.frame)
        pic.setGeometry(100, 10, 700, 490)
        pic.setPixmap(QPixmap("Draw.png"))

        tips = QLabel(self.frame)
        tips.setText("7 Questions")
        tips.setFont(QFont("Arial",14))
        tips.setGeometry(375, 530, 200, 50)    

    def __InitView(self):
        '''
        初始化界面
        '''

        self.frame.close()
        #新建一个水平布局作为本窗体的主布局
        main_layout = QHBoxLayout(self) 
        #设置主布局内边距以及控件间距为10px
        main_layout.setSpacing(10) 
    
        #在主界面左侧放置画板
        main_layout.addWidget(self.__paintBoard) 
        
        #新建垂直子布局用于放置按键
        sub_layout = QVBoxLayout() 
        
        #设置此子布局和内部控件的间距为10px
        sub_layout.setContentsMargins(10, 10, 10, 10) 

        self.__btn_Clear = QPushButton("清空畫板")
        self.__btn_Clear.setParent(self) #设置父对象为本界面
       
        #将按键按下信号与画板清空函数相关联
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear) 
        sub_layout.addWidget(self.__btn_Clear)
        
        self.__btn_Quit = QPushButton("退出")
        self.__btn_Quit.setParent(self) #设置父对象为本界面
        self.__btn_Quit.clicked.connect(self.Quit)
        sub_layout.addWidget(self.__btn_Quit)
        
        self.__cbtn_Eraser = QCheckBox("  使用橡皮擦")
        self.__cbtn_Eraser.setParent(self)
        self.__cbtn_Eraser.clicked.connect(self.on_cbtn_Eraser_clicked)
        sub_layout.addWidget(self.__cbtn_Eraser)


        self.next_question = QPushButton("")
        self.next_question.setParent(self)
        self.next_question.setFont(QFont("Arial",14))
        sub_layout.addWidget(self.next_question)

        
        splitter = QSplitter(self) #占位符
        sub_layout.addWidget(splitter)
        
        
        self.__question_label = QLabel("Question " + str(self.total_question))
        self.__question_label.setAlignment(Qt.AlignCenter) # 置中
        self.__question_label.setFont(QFont("Arial", 20)) # 字型、字體大小
        sub_layout.addWidget(self.__question_label)
       
        
        #把delphin拿來當空白
        self.i = random.randint(0,100)
        
        if(self.i == 27):
            self.i += 1
        
        self.appeared.append(self.i)
        self.__topic_label = QLabel(self.topics[self.i])
        self.__topic_label.setStyleSheet('background-color: rgb(255, 251, 100)')
        self.__topic_label.setText("<font color=%s>%s</font>" %('#227700', self.topics[self.i]))
        self.__topic_label.setAlignment(Qt.AlignCenter) # 置中
        self.__topic_label.setFont(QFont("Arial", 20)) # 字型、字體大小
        sub_layout.addWidget(self.__topic_label)


        self.__answer_label = QLabel("")
        self.__answer_label.setAlignment(Qt.AlignCenter) # 置中
        self.__answer_label.setFont(QFont("Arial", 20)) # 字型、字體大小
        sub_layout.addWidget(self.__answer_label)
        
        self.__message_label = QLabel("spend 0 s")
        self.__message_label.setAlignment(Qt.AlignCenter) # 置中
        self.__message_label.setFont(QFont("Arial", 20)) # 字型、字體大小
        sub_layout.addWidget(self.__message_label)
        
        self.__timeleft_label = QLabel("15 s")
        self.__timeleft_label.setAlignment(Qt.AlignCenter) # 置中
        self.__timeleft_label.setFont(QFont("Arial", 20)) # 字型、字體大小
        sub_layout.addWidget(self.__timeleft_label)
        
        main_layout.addLayout(sub_layout) #将子布局加入主布局
        
        self.__game_time()
        self.__recongnize()

    def __game_time(self):
        """ 遊戲計時 """
        
        self.game_timer.start(15500) # 一場遊戲15秒                
        #self.__message_label.setText("")
        self.game_timer.timeout.connect(self.__paintBoard.Clear) # 時間到清空畫板、換題目
        self.game_timer.timeout.connect(self.__add_total_sec)
        self.game_timer.timeout.connect(self.__change_topic)
        
        # 顯示剩下的秒數
        self.remain_timer.start(500)
        self.remain_timer.timeout.connect(self.__show_remain_time)    
    
    def __show_remain_time(self):
        #print(self.game_timer.remainingTime())
        self.__timeleft_label.setText(str(int(self.game_timer.remainingTime()/1000)) + " s")
    
    def __add_total_sec(self):
        """沒猜到答案 總費時加15秒"""
        self.score += 15

        self.__message_label.setText("spend "+str(self.score)+" s") #顯示猜對的訊息     

    
    def __stop_countdown(self):
        """ 猜到答案後，停止倒數 """
        self.hit_answer = True     
        self.score += 15 - int(self.game_timer.remainingTime()/1000)
        self.__message_label.setText("spend "+str(self.score)+" s") #顯示猜對的訊息    

        self.game_timer.stop()
        self.save_timer.stop()
        
        if(self.total_question < 7):
            self.next_question.setText("next")
            self.next_question.clicked.connect(self.__go_to_next_question)
        else:
            #self.next_question.setText("Game Over")
            self.__change_topic()

    def __go_to_next_question(self):
        if self.hit_answer == True:
            self.__paintBoard.Clear()
            self.__change_topic()
            
            self.game_timer.start(15500)
            self.remain_timer.start(500)
            self.save_timer.start(1000)

            self.next_question.setText("") 

        self.hit_answer = False
    
    def __recongnize(self):
        """ 每隔1秒就儲存圖片，才可以去分類 """
        self.save_timer = QTimer(self) # 要一直比對，所以利用QTimer定時將圖片傳給分類的函式
        self.save_timer.timeout.connect(self.on_btn_Save_Clicked)
        self.save_timer.start(1000) # 多久一次
    
    def __change_topic(self):
        
        if(self.total_question == 7):
            self.remain_timer.stop()
            self.__end_msg()
            pass
        
        self.total_question += 1
        
        self.i = random.randint(0,100)
        
        while(self.i in self.appeared or self.i == 27):
            self.i += 1
            
        self.appeared.append(self.i)
        self.__topic_label.setText("<font color=%s>%s</font>" %('#227700', self.topics[self.i]))
        self.__question_label.setText("Question " + str(self.total_question))
        self.__answer_label.setText("")
    
    def __end_msg(self):
        reply = QMessageBox.question(self, '資訊', '你花了 ' + str(self.score) + ' s\n再玩一次嗎？', \
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.__restart()
        else:
            self.Quit()

    def __restart(self):
        """ 再玩一次，要把所有Data初始化 """
        self.game_timer.start(15500)
        self.remain_timer.start(500)
        self.save_timer.start(1000)
        
        self.score = 0
        self.i = random.randint(0,100) #目前題目
        self.total_question = 0
        self.__message_label.setText("spend "+str(self.score)+" s")
        self.__paintBoard.Clear()
        self.appeared = []

    def on_btn_Save_Clicked(self):
        image = self.__paintBoard.GetContentAsQImage()
        image.save('draw\\your\\draw.jpg')
                
        self.guess()

    def guess(self):
        
        #開啟圖片 裁剪 設定大小
        img = PIL.Image.open("draw/your/draw.jpg")
    
        #img = img.resize((28,28),PIL.Image.ANTIALIAS)
    
        #反轉圖片顏色
        #print(str(img.mode))
        
        if img.mode == 'RGB':
            r,g,b = img.split()
            rgb_image = PIL.Image.merge('RGB', (r,g,b))
            inverted_image = PIL.ImageOps.invert(rgb_image)
            r2,g2,b2 = inverted_image.split()
            final_transparent_image = PIL.Image.merge('RGB', (r2,g2,b2))
            final_transparent_image.save('draw/your/draw.jpg')
        
        
    
    
        transform = transforms.Compose([
                    transforms.Resize((28,28)),
                    transforms.ToTensor()
                ])
        
        #讓電腦進行猜測
        test_data = torchvision.datasets.ImageFolder("draw",
                                                     transform = transform)
    
    
        test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                                  batch_size = 1,
                                                  shuffle = False)    
        
        for images, labels in test_loader:
            
            x = Variable(images.cuda(), requires_grad = False)
            #print(x.size())
        
            y = cnn(x) 
            #print(y)
            
            answer = torch.max(y, 1)[1]
        
            #print(answer.data)
            #print(self.topics[int(answer.data)])
        
        #跳過tent 把tent當作空白
        if self.topics[int(answer.data)] == "dolphin":
            self.__answer_label.setText("")
        else:
            self.__answer_label.setText("<font color=%s>%s</font>" %('#FF3333',self.topics[int(answer.data)]))
        
        
        if(self.topics[int(answer.data)] == self.topics[self.i]):
            self.__answer_label.setText("<font color=%s>%s</font>" %('#227700',self.topics[int(answer.data)]))

            #print(self.topics[int(answer.data)], self.topics[self.i])
            #print("猜到了")
            self.__stop_countdown()
        
        # 把圖片傳給分類的函式，就不用儲存到路徑 
        
    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True #进入橡皮擦模式
        else:
            self.__paintBoard.EraserMode = False #退出橡皮擦模式
        
    def Quit(self):
        self.close()

#建立模型
cnn = ResNet(ResidualBlock, [3,4,6,3])
cnn.cuda()
cnn.load_state_dict(torch.load("new_resnet34_params70.pkl"))





