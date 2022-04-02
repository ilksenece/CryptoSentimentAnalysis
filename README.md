# Project Name
> Outline a brief description of your project.
> Live demo [_here_](https://www.example.com). <!-- If you have the project hosted somewhere, include the link here. -->

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->


## General Information
“Bitcoin skyrocketed as much as 20% on Friday after Elon Musk added the hashtag #bitcoin to his Twitter bio.
The virtual currency climbed suddenly at around 3:30 a.m. ET, adding $5,000 in the space of an hour to trade at $37,299, according to data from industry site CoinDesk. At 8:00 a.m. ET, it extended gains to trade around $37,653.” states CNBC article from Jan 29th of 2021 showcasing how sensitive the digital currency’s trading value is to comments posted on social media platforms like Twitter by influential accounts like Musk’s. Inspired by phenomena as such that we have all witnessed over the course of past few years, it would be really powerful to be able to perform the sentiment analysis on tweets on a particular day/hour containing #bitcoin or #btc and predict the pertaining leading sentiment regarding the crypto currency and maybe review one’s investment decisions accordingly. With this motivation, for the Capstone project, I am interested in solving the underlying prediction problem. 
This idea, not surprisingly, has already been explored in the ML community. I managed to solve labeled tweets from the last few years that could be used to train models. Also Twitter API enables users to pull a few hundred thousands of tweets monthly. A manual labeling might also be performed if need be for additional data for training and/or testing purposes. 
This is going to be a supervised classification problem where popular tweets will be ultimately classified as “positive”, “neutral” or “negative”. This is an instance of NLP, traditionally neural nets are being used for language comprehension purposes. However due to the high cost of the training process of the task in hand, for our purposes, we can leverage a pre-trained model developed for a similar source setting and apply it to our specific target setting. In other words, we plan to also evaluate deep transfer learning as a viable approach in this project. 


## Technologies Used
- Transfer Learning 
- Hugging Face: DistilBERT: https://huggingface.co/docs/transformers/model_doc/distilbert
- Google Cloud to create the container for the image and deploy the application: https://v0-0-1-kzw4svp5zq-uc.a.run.app/


## Features
- Returns the sentiment prediction (positive/negative/neutral) associated with Cryptocurrency related tweets


## Screenshots
![Example screenshot1](./img/screenshot1.jpg)
![Example screenshot2](./img/screenshot2.jpg)
<!-- If you have screenshots you'd like to share, include them here. -->


## Setup
Project requirements are listed in requirement.txt file in the directory. 



## Project Status
Project is currently not being worked but definitely there is always room for more.


## Room for Improvement

Room for improvement:
- The training/testing data that has been used is pulled from Keggle. Even after training the DistilBert Model, although the accuracy on testing set seems to be pretty high, manual tests prove that the model is not providing high accuracy in predicting sentiments associated with tweet sentences related to Cryptocurrency. A new data set needs to be collected and labeled and DistilBert model needs to be retrained.

To do:
- A continuous tweet stream can be fed to model and monitored for frequent automated training.


## Acknowledgements
- This project was inspired by my Springboard mentor Zuraiz Uddin.
- This project was a Capstone project for UCSD Extension Machine Learning Engineering bootcamp offered by Springboard.
- Many thanks to my mentor, my family.


## Contact
Created by [@ilksenece](https://github.com/ilksenece/CryptoSentimentAnalysis) - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
