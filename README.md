# Automated Email Chatbot
Name - Lokesh Khandelwal<br>
Roll no. - 21110113<br>
Institue - IIT Gandhinagar<br>

## Architecture:

![Project screenshot](img/chatbot%20architecture.jpg)

The basic flow of my application is like following:<br>
Step 1: User gives input to the chatbot.<br>
Step 2: The input is given to the email-classifer, which classifies the image into three classes (students, researcher, corporates).<br>
Step 3: If the class is 'corporates' the mail saying the email has been forwarded to HOD is sent.<br>
Step 4: If not corporate, the mail is sent to VectorDB where it retrieves the most relevant document asked in the mail using similarity search.<br>
Step 5: The email, the class type(student, researcher), the retrieved documents are sent as an input to LLama3 which is prompted with following specifications:<br>
        - If it is a mail having any sensitive information, reply that the mail is sent to dean/Hod.<br>
        - If the mail asks for any other general query, LLama3 uses the retrieved documents to draft an email.<br>
        - if the question asked is not in documnets, it replies that it has been sent to higher ups, you will get a reply soon.<br>

## Instructions:<br>

To get Groq-API-KEY: Go to groq.com -> Devlopers section -> API Keys -> Create new API Key for free and replace it.<br>
For pretrained model: please download the model from the following link:<br>


https://iitgnacin-my.sharepoint.com/:f:/g/personal/21110113_iitgn_ac_in/EiQ2SA5z__hBscIWWi3OLYEBz_sI8l8d7pkkpuK5ZZCllQ?e=V1FLkp<br>
Replace the model path in model_path variable in src/email-classifier-pretrained.py file from your path.<br>

Run the src/app.py file using streamlit run app.py<br>
```bash
streamlit run app.py
```

## VectorDB Used:<br>
I have used ChromaDB as my db to store documents embeddings, due to its ease of use with langchain, it works well with langchain libraries and has many type of uploaders. I have uploaded the file chroma-rag-files/chroma_upload.py which uploades pdf as embedding in chunks to the db.<br>

## Classifier Used:<br>
I have fine-tuned the RobertA model for the email classification task. The main reason behind selecting it, is it's transformer based architecture. RoBERTa's fine-tuning capabilities allow it to map input texts to multiple categories effectively, while its deep layers enable it to distinguish subtle differences between classes. Overall, RoBERTa’s state-of-the-art performance and strong generalization make it ideal for multiclass classification.<br>

## LLamma3-70b-8192:<br>
I used LLama3-70b-8192 for the dataset generation as well as for drafting mails. I prompted the model according to the usecase. Also in the final layer, it is prompted to identify any sensitive information in the mail and draft the mail accordingly.<br>
        
## Working of Chatbot<br>


