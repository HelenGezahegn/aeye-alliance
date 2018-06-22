# AEyeAlliance
AI4SocialGood Final Project: Braille to text converter (Optical Character Recognition/Optical Braille Recognition)

## About us ##
We are a group of four undergraduate female students in computer science who are currently partaking in the AI4SocialGood Lab at McGill University. Throughout the first 2 weeks of the lab, we learned about AI and machine learning techniques and are now applying this knowledge to solve a social problem.
![alt text](https://github.com/HelenG123/aeye-alliance/blob/master/static/AEyeAlliance.jpg)

## Introduction ##
Based on personal experience, we felt that there weren't enough technology out there to help non-blind people interact and communicate with blind people. So, we decided to create a braille-to-text converter that is able to take an image of a line (a word, sentence, or a character) of Braille, recognize it, and output its English counterpart. Using data acquired from rare book collections and the internet, we constructed our own dataset consisting of multiple sets of the Braille alphabet, numbers, and symbols. We then trained an AI that will recognize images of individual Braille characters and translate them into English characters. Using a specific ratio and cropping function, we are able to take an image of a sentence/line of Braille and segment each Braille character out individually. We feed each of these characters into our model and then concatenate the output characters together into an English sentence or word.

Our model is able to recognize all the letters in the alphabet, capital letters, numbers, commas, periods, spaces, hyphens, apostrophes, colons, semicolons, question marks, and exclamation marks. Please refer to the chart below. 

![Braille Character Chart](https://github.com/HelenG123/aeye-alliance/blob/master/static/braille_character_chart.jpg)


The main objective of this project is to bring awareness of Braille and to bridge the gap of communication between blind and non-blind people.

## User Interfaces ##
* website
* android app

Unfortunately, due to error with some kernel operations from using tensorflow on Android Studio, we were not able to successfully import and use our model on our android app. [We have posted this error on StackOverFlow.](https://stackoverflow.com/questions/50955816/java-lang-illegalargumentexception-no-opkernel-was-registered-to-support-op-ga) For the time being, we will use our website as the main Braille-to-Text converter.

## Open Source Dataset ##
Since we were not able to find an open source or public dataset of images of English Grade 1 Braille, we are planning on releasing our own dataset to be downloaded and used for free. 

## How to run flask to access our website ##
1. Make sure your machine has Flask installed in either conda or pip
2. Clone/Download this repository
3. Navigate to the repo directory on your command prompt and type
`python learn_flask.py`
4. You should see something similar to this
![running flask](https://github.com/HelenG123/aeye-alliance/blob/master/flask.png)
5. If you get an "ImportError: No module named flask_uploads", try
`pip install Flask_Uploads`
6. Copy and paste the HTTP link onto your browser. 
7. If you are unsure about how to use our Braille-to-Text converter, there is a Tutorial page on our site that will show you how. 

## Credits ##
Special thanks to our mentors Peng Yu and Francis Gregoire!

