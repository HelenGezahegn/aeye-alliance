# AEyeAlliance
AI4SocialGood Final Project: Braille to text converter (Optical Character Recognition/Optical Braille Recognition)

We are a group of four undergraduate female students in computer science who are currently partaking in the AI4SocialGood Lab at McGill University. Throughout the first 2 weeks of the lab, we learned about AI and machine learning techniques and are now applying this knowledge to solve a social problem.

Based on personal experience, we felt that there weren't enough technology out there to help non-blind people interact and communicate with blind people. So, we decided to create a braille-to-text converter that is able to take an image of a line (a word, sentence, or a character) of Braille, recognize it, and output its English counterpart. Using data acquired from rare book collections and the internet, we constructed our own dataset consisting of multiple sets of the Braille alphabet, numbers, and symbols. We then trained an AI that will recognize images of individual Braille characters and translate them into English characters. Using a specific ratio and cropping function, we are able to take an image of a sentence/line of Braille and segment each Braille character out individually. We feed each of these characters into our model and then concatenate the output characters together into an English sentence or word.

Our model is able to recognize all the letters in the alphabet, capital letters, numbers, commas, periods, spaces, hyphens, apostrophes, colons, semicolons, question marks, and exclamation marks. Please refer to the chart below. 

![Braille Character Chart](https://github.com/HelenG123/aeye-alliance/blob/master/static/braille_character_chart.jpg)


The main objective of this project is to bring awareness of Braille and to bridge the gap of communication between blind and non-blind people.

Special thanks to our mentors Peng Yu and Francis Gregoire!

