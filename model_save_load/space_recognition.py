import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image

# from autocorrect import spell


def make_prediction(img_path):
    model = CNN()
    model.load_state_dict(torch.load("current_model_7.pth"))
    image = Image.open(img_path)
    image = image.convert('RGB')
    width, height = image.size
    num = round(width/height/0.78)
    w = width/num

    letters = []
    for i in range(0, num):
        cropped = image.crop((i * w, 0, (i + 1) * w, height))
        # cropped.show()
        cropped = np.array(cropped)
        cropped = cv2.resize(cropped, (28, 28))
        cropped = cropped.astype(np.float32) / 255.0
        cropped = torch.from_numpy(cropped[None, :, :, :])
        cropped = cropped.permute(0, 3, 1, 2)
        predicted_tensor = model(cropped)
        _, predicted_letter = torch.max(predicted_tensor, 1)
        if int(predicted_letter) == 26:
            letters.append(chr(32))
        elif int(predicted_letter) == 27:
            letters.append(chr(35))
        elif int(predicted_letter) == 28:
            letters.append(chr(46))
        elif int(predicted_letter) == 29:
            letters.append(chr(44))
        elif int(predicted_letter) == 30:
            letters.append(chr(58))
        elif int(predicted_letter) == 31:
            letters.append(chr(92))
        elif int(predicted_letter) == 32:
            letters.append(chr(45))
        elif int(predicted_letter) == 33:
            letters.append(chr(59))
        elif int(predicted_letter) == 34:
            letters.append(chr(63))
        elif int(predicted_letter) == 35:
            letters.append(chr(33))
        elif int(predicted_letter) == 36:
            letters.append(chr(126))
        else:
            letters.append(chr(97 + predicted_letter))

    output = ""
    number = False
    capL = False
    capW = False
    for j in letters:
        if j == '#':
            number = True
        elif ord(j) == 126:
            if capL:
                capW = True
            capL = True
        elif j == ' ':
            number = False
            capL = False
            capW = False
            output = output + j
        elif not number:
            if not capL or not capW or ord(j) not in range(97, 123):
                output = output + j
            else:
                output = output + chr(ord(j)-32)
                capL = False
        else:
            if ord(j) in range(97, 106):
                output = output + chr(ord(j)-48)
            elif ord(j) == 106:
                output = output + char(48)
            else:
                output = output + j

    # return spell(str(output))
    return output


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(
            # 3x28x28
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            # 16x28x28
            nn.MaxPool2d(kernel_size=2),
            # 16x14x14
            nn.LeakyReLU()
        )
        # 16x14x14
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            # 32x14x14
            nn.MaxPool2d(kernel_size=2),
            # 32x7x7
            nn.LeakyReLU()
        )
        # linearly
        self.block3 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 37)
        )
        # 1x36

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        # flatten the dataset
        out = out.view(-1, 32 * 7 * 7)
        out = self.block3(out)

        return out

print(make_prediction("afternoon_with.png"))
print(make_prediction("brought_the_ball_to.png"))
print(make_prediction("threw_the_ball.png"))
print(make_prediction("little_girl.png"))
print(make_prediction("with_his_family.png"))
print(make_prediction("the_little.png"))
print(make_prediction("the_daddy.png"))
print(make_prediction("and_laughed.png"))
print(make_prediction("would_run_and_get_it.png"))
print(make_prediction("he_took_it_home_to_play.png"))
print(make_prediction("picked_it_up_with_his_mouth.png"))

# print(make_prediction("family.jpg"))
# print(make_prediction("home.jpg"))
# print(make_prediction("took.jpg"))
# print(make_prediction("16.png"))
# print(make_prediction("30.png"))
# print(make_prediction("53.png"))
# print(make_prediction("76.png"))
# print(make_prediction("86.png"))
# print(make_prediction("124.png"))
#
# print(make_prediction("na--in-.jpg"))
# print(make_prediction("says,.jpg"))
# print(make_prediction("sp,k.jpg"))
# print(make_prediction("1926..jpg"))