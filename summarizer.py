from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI, OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


import os
import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get('OPENAI_API_KEY')

target_len = 500
chunk_size = 3000
chunk_overlap = 200

tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="fr")
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")



def summarize(text, lang):
    try:
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts[:]]
        openaichat = OpenAIChat(temperature=0, model="gpt-3.5-turbo")
        prompt_template = """Act as a professional technical meeting minutes writer.
        Tone: formal
        Format: Technical meeting summary
        Length:  200 ~ 300
        Tasks:
        - highlight action items and owners
        - highlight the agreements
        - Use bullet points if needed
        {text}
        CONCISE SUMMARY IN ENGLISH:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        f"Given the new context, refine the original summary in English within {target_len} words: following the format"
        "Participants: <participants>"
        "Discussed: <Discussed-items>"
        "Follow-up actions: <a-list-of-follow-up-actions-with-owner-names>"
        "If the context isn't useful, return the original summary. Highlight agreements and follow-up actions and owners."
        )
        refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
        )
        chain = load_summarize_chain(
                openaichat,
                chain_type="refine",
                return_intermediate_steps=True,
                question_prompt=PROMPT,
                refine_prompt=refine_prompt,
            )
        
        resp = chain({"input_documents": docs}, return_only_outputs=True)
        print(resp["output_text"])
        # return resp["output_text"]
    
        text = resp['output_text']
        encoded_zh = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id(lang))
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    except:
        pass