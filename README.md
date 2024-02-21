# Docs QA Bot🤖

A streamlit app that enables users to interact with the uploaded PDF. You can ask questions or doubts regarding the PDF and our Chatbot would answer them with a friendly response.

<img src="imgs\docsearch.png" alt="alt text" width="800" height="400">

## Tech stack

* 🐍Python
* 🛑🔥Streamlit
* 🦜️🔗Langchain
* 🔰FAISS Vectorstore
* ❇️Google Generative AI
* 🆚Git & Github

## Working
Let's breakdown the working of the app into chunks to make it easier to understand:

* Upload the PDF
* Extract the text from the PDF file
* Generate embeddings of the text
* Store the embeddings in the vectorstore
* Retrieve the closest match
* Display the results in a Chatbot (Interface)

## Display the results in a Chatbot (Interface)
![alt text](imgs\app_screenshot2.png)