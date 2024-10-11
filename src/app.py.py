import warnings
warnings.filterwarnings('ignore')
import torch
from transformers import RobertaModel
import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from email_classifier_pretrained import classify_email
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

groq_api_key = os.environ.get("GROQ_API_KEY")
model = 'llama3-70b-8192'

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

client = chromadb.PersistentClient(path="/mnt/c/Users/lokes/Desktop/Smartsense/chroma_embedding")
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
)
db = Chroma(client=client, embedding_function=embeddings)
retriever = db.as_retriever()
memory = ConversationBufferMemory()

groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model,
    )

def main():
    st.title("Email Assistant Chatbot")

    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []

    with st.form(key='input_form', clear_on_submit=True):
        user_input = st.text_input("You:")
        submit_button = st.form_submit_button(label='Send')

    if submit_button:
        if user_input:
            st.session_state['conversation_history'].append(("User", user_input))
            output = process_message(user_input)
            st.session_state['conversation_history'].append(("Bot", output))

    for speaker, msg in st.session_state['conversation_history']:
        if speaker == "User":
            st.markdown(f"**{speaker}:** {msg}")
        else:
            st.markdown(f"**{speaker}:** {msg}")

def process_message(message_content):
    sender = classify_email(message_content)
    output = ""
    try:
        if sender == "corporates":
            output = "Your mail has been sent to the respective HOD. You will get a reply soon. Have a good day!"
        else:
            relevant_docs = db.similarity_search(message_content)

            system_prompt2 = (
                "You are a helpful email assistant who helps in sending replies for the email based on the given scenarios and conditions. "
                "Scenarios: "
                "1. In input you will be given the Email body, who the sender's class is (student/researcher) and any relevant document required in replying the email. You will draft the response accordingly. "
                "2. You will then check if the email contains any sensitive information (like theft or harassment issues) or any other personal or serious information. If it contains any of the mentioned types of information, then you will just draft an email saying that the mail has been sent to the HOD/Dean and your issue will be addressed soon. "
                "3. If the email asks for any general query, you will draft the mail based on the documents and class type you are provided. "
                "4. If the document does not have information to answer the question, just output 'I currently don't have enough information on it; your mail has been sent to higher-ups.'"
            )

            prompt2 = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt2),
                    HumanMessagePromptTemplate.from_template("{human_input}"),
                ]
            )

            conversation2 = LLMChain(
                llm=groq_chat,
                prompt=prompt2,
                verbose=False,
            )

            output = conversation2.predict(
                human_input=f"Email Content:\n{message_content}\n\nClass(person who sent the email): {sender}\n\nRelevant Information:\n{relevant_docs}"
            )
            print(output)
        return output
    except Exception as e:
        output_ba = "This currently exceeds my token limit. Can I assist you with something else?"
        print(e)
        return output_ba

if __name__ == "__main__":
    main()
