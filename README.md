# CS535---COVID-19-Prediction
Big Data Course Project - COVID-19 Prediction

The COVID-19 pandemic is the first time in history that we have experienced a problem of this size, and at the same
time, we can process and understand the large amount of data gathered. We have various organizations (government,
newspapers, academic institutions, etc.) collecting and publishing COVID-19 related data continuously. We need a
robust and quick method to process substantial amounts of data and display meaningful and actionable valuable output
for the public's health.
Using the Google Open dataset, we process infectious and morbidity rates based on population size to analyze the
results of 3 different time series models. Counties in the US are identified as rural, suburban, and urban for comparison
purposes. We have used three deep learning models – Stacked LSTM, Bidirectional LSTM, and Multivariate CNN to
perform comparative analysis. Bidirectional LSTM shows the best performance for rural and suburban regions,
whereas Stacked LSTM shows the best performance for urban areas. The accuracy of these models and the speed at
which they process the results will guide us in deciding which model works best for the given population size. We
also present a visual summary of the results. Using Big Data techniques with distributed models is the straightforward
way to process the massive amounts of data. Utilizing a cluster with more than one worker will enable the daily
processing of the incoming data quickly. 
