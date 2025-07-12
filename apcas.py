"""
                    APCAS 2.0 (Any PDF Chatting AI System)
                    =========================================================================================================

                    VERSION = 2.0

                    APCAS stands for Any PDF Chatting AI System, it is basically a RAG (Retrieval Augmented Generation) based
                    application, that optimized for chatting with any PDF in an efficint way, still it is not a professional 
                    project So it may have several bugs.


                    Features:
                        - It provide chatting on Images
                        - It extract all images from PDF

                    
                    Limitations:
                        - It takes time when generating summary of each image (internally).
                        - Increasing in number of images directly leads more time consumption.

                    Example:
                        ```
                            from apcas import APCAS
                            model = APCAS('path_of_pdf')
                            model.run_on_terminal()
                        ```
                            
"""











#------------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------------
import os
import warnings
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
import fitz  # PyMuPDF for PDF handling
from base64 import b64encode
from typing import Literal, List, Dict, Tuple, NoReturn, Self, Optional
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage








#------------------------------------------------------------------------------------
# Configurations
#------------------------------------------------------------------------------------

MODEL_NAME = "APCAS"
VERSION = '2.0'
CORE_LLM = "GPT 4o mini"












########################################################################################################
# 
#                                               Main class (APCAS 2.0)
# 
########################################################################################################

class APCAS_2_0(Runnable):
    ''' Main class '''


    __name__    = "APCAS 2.0(Any PDF Chatting AI System)"
    __model__   = 'GPT 4o mini'
    __version__ = VERSION # 2.0





    def __init__(
            self, 
            pdf_path: Optional[str] = None,
            verbose: bool = True,
            show_warnings: bool = True
        
        ) -> Self:
        """PDFImagesChattingRAG:
            This is the main class for PDF's Images Chatting AI System \
            using RAG (Retriever Augmented System).

        Returns:
            Self: It's own class instance.
        """

        # loading secret variables
        load_dotenv()

        # Validating PDF path manually
        if not os.path.exists(pdf_path):
            raise FileNotFoundError("Your provided file is not exists.")
        
        # deleting all pre-existing images
        self._clear_images_folder()

        ## object instances - declaration and some functions calling
        self.total_images = 0
        self.verbose: bool = verbose
        self.pdf_path: str = pdf_path
        self.show_warning: bool = show_warnings
        if self.pdf_path: # process pdf if user give path
            self.process_pdf()

        self.azure_client = self.load_azure_client()
        self.embedding_model = self.load_embedding_model()

        # calling vector store function
        self.save_vector_store()


        # checking is 'images/' folder exists or not
        if not os.path.exists("images"):
            # displaying some crucial warnings
            if self.show_warning:
                warnings.warn("Before further moving forward, must be create an folder (images) in this directory.")

        






    #------------------------------------------------------------------------------------
    # Function (private): delete all previous images
    #------------------------------------------------------------------------------------
    def _clear_images_folder(self) -> NoReturn:
        """ This function will delete all pre-existing files/images """

        for file in os.listdir(folder:="images"):
            os.remove(os.path.join(folder,file))

        return True






    #------------------------------------------------------------------------------------
    # Function (Private): Convert PDF page into image(png)
    #------------------------------------------------------------------------------------
    
    def _pdf_page_to_png(self, doc: fitz.open, doc_id: Optional[str] = "doc_01") -> True:
        """This function save a doc (page of PDF get by calling fitz.open function) as .png image

        Args:
            doc (fitz.open): loaded page using fitz
            doc_id (Optional[str], optional): name of image to be saved. Defaults to "doc_01".

        Returns:
            True: when everything done
        """

        # Render page to pixmap (image)
        pix = doc.get_pixmap()
        
        # Save pixmap as PNG
        pix.save(f"images/{doc_id}.png")
        
        return True








    #------------------------------------------------------------------------------------
    # (Class) Function: Update PDF path
    #------------------------------------------------------------------------------------

    def update_pdf_path(self, pdf_path: str) -> NoReturn:
        """Call this function to update the PDF path

        Args:
            pdf_path (str): Actual path in the memory (which is saved)

        Returns:
            NoReturn: Returns True, and it will update PDF path only.
        """
        
        # Validating PDF path manually
        if not os.path.exists(pdf_path):
            raise FileNotFoundError("Your provided file is not exists.")
        
        self.pdf_path = pdf_path

        return True






    #------------------------------------------------------------------------------------
    # (Class) Function: Load Azure Client
    #------------------------------------------------------------------------------------

    def load_azure_client(self) -> AzureOpenAI:
        """ This function will load all the credentials and setup GPT-4o-mini LLM model
            afterthat return as instance of Azure Open AI class

        Returns:
            AzureOpenAI: Instance of class (main execution) 
        """ 

        client = AzureOpenAI()

        self.azure_client = client

        if self.verbose:
            print("Azure Client Loaded successfully.")

        return client





    #------------------------------------------------------------------------------------
    # (Class) Function: Load Embedding Model
    #------------------------------------------------------------------------------------
    def load_embedding_model(self, model_name: Optional[str] = "text-embedding-ada-002") -> AzureOpenAIEmbeddings:
        """This function load Embedding model and autoset its credentials.

        Args:
            model_name (str): Describe the embedding model name of OpenAI using Azure
        Returns:
            AzureOpenAIEmbeddings: It will returns object of embedding model of OpenAI (using Azure)
        """

        # Initialize embeddings for summaries
        embedding_model = AzureOpenAIEmbeddings(
            model=model_name
        )

        if self.verbose:
            print("Embedding model loaded successfully.")

        return embedding_model





    #------------------------------------------------------------------------------------
    # (Class) Function: Process PDF
    #------------------------------------------------------------------------------------

    def process_pdf(self, pdf_path: Optional[str] = None) -> NoReturn:
        """Load PDF and extract all available images from it. Must 

        Args:
            pdf_path (Optional[str], optional): Path of PDF. Defaults to None.

        Returns:
            NoReturn: returns True when everything processed successfully
        """

        if not pdf_path:
            pdf_path = self.pdf_path

            if not pdf_path:
                raise FileNotFoundError("Provide path of PDF first")
            
        images_count = 0                                        # creating instance to store how much images saved yet
        doc = fitz.open(pdf_path)                               # loading pdf
        if len(doc) == 1:                                       # condition if pdf contain single page
            self._pdf_page_to_png(doc[0])
            self.total_images = 1
            return True
        
        for page_num, page in enumerate(doc):                   # iterating each page
            page_num += 1
            images = page.get_images(full=True)                 # extracting image(s) in current page
            for img_index, img in enumerate(images):            # iterating each image if more than one in same page
                xref = img[0]
                base_image = doc.extract_image(xref)
                doc_id = f"doc_{page_num}_{img_index}"          # generating unique doc_id
                with open(f"images/{doc_id}.png", "wb") as f:   # loading current image (binary data)
                    f.write(base_image["image"])                # saving current image
                    images_count += 1
                    if self.verbose:
                        print(f"Image extracted and saved to directory (.../images/{doc_id}).")


        self.total_images = images_count

        if images_count == 0:
            if self.verbose:
                print("No image extract.")
            return False
        
        else:
            return True









    #------------------------------------------------------------------------------------
    # (Class) Function: Generate summary of each image
    #------------------------------------------------------------------------------------
    def generate_summary_of_each_image(
            self,
            extracted_images_folder_path: str, 
            verbose: bool = True
            
            ) -> Tuple[AzureOpenAI.chat, Dict[str, str], Dict[str, str]]:
        
        """ This function takes the path of folder in which extracted images are stores,
            and generate the summary of each image by the use of LLM, finally returned it.

        Args:
            extracted_images_folder_path (str): Path of folder that contains extracted images from PDF

        Returns:
            Dict[str, str]: Returns a python dictionary that contains the unique doc_id as key as 
                            generated summary of images as value
        """

        # checking provided folder path exists or not
        if not os.path.exists(extracted_images_folder_path):
            raise FileNotFoundError(f"Folder doesn't exists: {extracted_images_folder_path}. Try to provide full folder path.")

        # generating embeddings of summaries
        embed_summaries: Dict[str,str] = {}
        generated_summary: Dict[str,str] = dict()                               # defining variable which store generated summaries of each image
        images_path: List[str] = os.listdir(extracted_images_folder_path)       # fetching all images paths
        total_images_count = len(images_path)                                   # extracting total number of images

        ## iteration of each image
        for index,image_path in enumerate(images_path):
            doc_id = image_path.split('.')[0]                                   # extracting doc_id from image title
            
            # updating image path with its parent folder
            image_full_path = os.path.join(extracted_images_folder_path, image_path)
                                
            # fetching image data
            with open(image_full_path, "rb") as f:
                image_b64 = b64encode(f.read()).decode("utf-8")

            if verbose:
                print(f"Generating summary of image ({index+1}/{total_images_count}).... ")


            data_url = f"data:image/png;base64,{image_b64}"                     # creating data url variable

            response = self.azure_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert vision assistant."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Generate a detailed summary for the given image only. Making sure you summarize everything what you see in the image."},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=1.0,
                top_p=1.0
            )

            summary: str = response.choices[0].message.content                  # fetching summary from llm response
            summary_embedding = self.embedding_model.embed_query(summary)       # getting embedding of generated summary
            generated_summary.update({doc_id:summary})                          # add generated summary to summaries
            embed_summaries.update({doc_id:summary_embedding})                  # add summary embedding to summary embeddings

            

        return response, generated_summary, embed_summaries










    #------------------------------------------------------------------------------------
    # (Class) Function: Saving Embeddings of Summaries to vector Store
    #------------------------------------------------------------------------------------

    def save_vector_store(self) -> NoReturn:
        """This function built vector store and save it

        Returns:
            NoReturn: True when vector stored successfully.
        """

        if self.total_images == 1:
            return True

        # getting summaries
        _,_,embed_summaries = self.generate_summary_of_each_image("images")

        if not embed_summaries:
            vector_store = FAISS.from_texts(['No-Images found'], embedding=self.embedding_model)
        else:
            for key, embed_summary in embed_summaries.items():
                doc_id = key
                vector_store = FAISS.from_embeddings(
                    [(doc_id, embed_summary)],
                    embedding=self.embedding_model
                )

        vector_store.save_local("faiss_index")

        return True







    #------------------------------------------------------------------------------------
    # (Class) Function: Query System]
    #------------------------------------------------------------------------------------

    # Step 7-9: Handle user query and display response
    def query_system(self, user_query: str) -> str:
        """ Setting up all the credentials and making sure system working currectly.

        Args:
            user_query (str): query of user 

        Returns:
            str: response from LLM
        """

        if self.total_images == 0:
            return "No images detected (This problem occurs in system)"

        if self.total_images == 1:
            doc_id = os.listdir("images")[0].split('.')[0]

        else:
            vector_store = FAISS.load_local(
                folder_path = "faiss_index", 
                embeddings = self.embedding_model,
                allow_dangerous_deserialization = True
            )

            # creating retriever
            retriver = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.00000001}
            )

            # fetching appropriate docs from retriever
            results = retriver.invoke(user_query)

            if not results: # if no docs fetched from retriever
                print("I don't know (No image applicable for your this query)")
                return
            
            doc_id = results[0].page_content                        # extracting image unique doc_id

            
        image_path = f"images/{doc_id}.png"                     # fetching image from doc_id

        with open(image_path, "rb") as f:                       # fetching image data
            image_b64 = b64encode(f.read()).decode("utf-8")     # Encoding image binary in UTF-8

        data_url = f"data:image/png;base64,{image_b64}"         # creating data_url from image data
        
        # Step 8: Pass image to LLM with prompt
        response = self.azure_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert vision assistant. Don't use phrases like 'In the context of image...' or its relevant."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Answer the query in the context of provided image. Query: {user_query}"},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ],
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0
        )

        # Step 9: Display response
        return response.choices[0].message.content






    #------------------------------------------------------------------------------------
    # (Class) Function: Invoke
    #------------------------------------------------------------------------------------
    def invoke(self, query: str) -> AIMessage:
        """Invoke Function 

        Args:
            input (str): Question of user

        Returns:
            AIMessage: AI Message which contains the response of AI
        """

        response = self.query_system(user_query=query)

        return AIMessage(content=response)







    #------------------------------------------------------------------------------------
    # (Class) Function: Run on Terminal
    #------------------------------------------------------------------------------------
    def run_on_terminal(self) -> NoReturn:
        """ Activate Chat AI feature in your terminal """

        print(f"{'-'*50} APCAS ({self.__version__}) has been activated! Now you can chat with it! {'-'*50}")

        while True:

            query = input("\n\n\033[33m-----(YOU): ").lower().strip()
            if query == "":
                continue
            if query in ["exit","bye","bye bye"]:
                print("\n\n\tBye Bye!\n\n")
                break

            response = self.invoke(query)

            # displaying ouptut
            print("\n\n\033[32m-----(AI Response)\n\t|")
            for line in response.split('\n'):
                print(f"\t| {line}")
            print("\t|")
    








if __name__ == "__main__":

    ai = APCAS_2_0(pdf_path="content/attention.pdf")

    # ai.run_on_terminal()

    

    













