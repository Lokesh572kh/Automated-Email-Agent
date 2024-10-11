# Email Handler
Name - Lokesh Khandelwal
Roll no. - 21110113
Institue - IIT, Gandhinagar


The basic flow of my application is like following:
Step 1: User gives input to the chatbot.
Step 2: The input is given to the email-classifer, which classifies the image into three classes (students, researcher, corporates).
Step 3: If the class is 'corporates' the mail saying the email has been forwarded to HOD is sent.
Step 4: If not corporate, the mail is sent to VectorDB where it retrieves the most relevant document asked in the mail using similarity search.
Step 5: The email, the class type(student, researcher), the retrieved documents are sent as an input to LLama3 which is prompted with following specifications:
        - If it is a mail having any sensitive information, reply that the mail is sent to dean/Hod.
        - If the mail asks for any other general query, LLama3 uses the retrieved documents to draft an email.
        - if the question asked is not in documnets, it replies that it has been sent to higher ups, you will get a reply soon.l


VectorDB Used:
I have used ChromaDB as my db to store documents embeddings, due to its ease of use with langchain, it works well with langchain libraries and has many type of uploaders. I have uploaded the file chroma-rag-files/chroma_upload.py which uploades pdf as embedding in chunks to the db.


        




