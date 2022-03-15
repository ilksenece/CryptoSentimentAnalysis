“Bitcoin skyrocketed as much as 20% on Friday after Elon Musk added the hashtag #bitcoin to his Twitter bio.
The virtual currency climbed suddenly at around 3:30 a.m. ET, adding $5,000 in the space of an hour to trade at $37,299, according to data from industry site CoinDesk. At 8:00 a.m. ET, it extended gains to trade around $37,653.” states CNBC article from Jan 29th of 2021 showcasing how sensitive the digital currency’s trading value is to comments posted on social media platforms like Twitter by influential accounts like Musk’s. Inspired by phenomena as such that we have all witnessed over the course of past few years, it would be really powerful to be able to perform the sentiment analysis on tweets on a particular day/hour containing #bitcoin or #btc and predict the pertaining leading sentiment regarding the crypto currency and maybe review one’s investment decisions accordingly. With this motivation, for the Capstone project, I am interested in solving the underlying prediction problem. 
This idea, not surprisingly, has already been explored in the ML community. I managed to solve labeled tweets from the last few years that could be used to train models. Also Twitter API enables users to pull a few hundred thousands of tweets monthly. A manual labeling might also be performed if need be for additional data for training and/or testing purposes. 
This is going to be a supervised classification problem where popular tweets will be ultimately classified as “positive”, “neutral” or “negative”. This is an instance of NLP, traditionally neural nets are being used for language comprehension purposes. However due to the high cost of the training process of the task in hand, for our purposes, we can leverage a pre-trained model developed for a similar source setting and apply it to our specific target setting. In other words, we plan to also evaluate deep transfer learning as a viable approach in this project. 