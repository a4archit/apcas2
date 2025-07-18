

#------------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------------
import os
import warnings
import fitz  # PyMuPDF for PDF handling
import imagehash
import io 

from PIL import Image
from base64 import b64encode
from typing import Literal, List, Dict, Tuple, NoReturn,  Optional





#------------------------------------------------------------------------------------
# Function (Private): Compare images and say find image unique or not
#------------------------------------------------------------------------------------
def _are_images_identical(img, img_path2: str):
    """
    Compare two images to determine if they are exactly the same.
    
    Args:
        img_path1 (str): File path to the first image.
        img_path2 (str): File path to the second image.
    
    Returns:
        bool: True if images are exactly identical, False otherwise.
    """
    try:
        # Open images using PIL
        # img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')
        
        # Calculate perceptual hashes
        hash1 = imagehash.average_hash(img, hash_size=8)
        hash2 = imagehash.average_hash(img2, hash_size=8)
        
        # Check for exact match (hash difference == 0)
        return hash1 == hash2
    except Exception as e:
        print(f"Error comparing images: {e}")
        return False





#------------------------------------------------------------------------------------
# Function (private): delete all previous images
#------------------------------------------------------------------------------------
def _clear_images_folder(folder: Optional[str] = "extracted_images") -> NoReturn:
    """ This function will delete all pre-existing files/extracted_images """

    for file in os.listdir(folder):
        os.remove(os.path.join(folder,file))

    return True






#------------------------------------------------------------------------------------
# Function (Private): Convert PDF page into image(png)
#------------------------------------------------------------------------------------

def _pdf_page_to_png(doc: fitz.open, doc_id: Optional[str] = "doc_01") -> True:
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
    pix.save(f"extracted_images/{doc_id}.png")
    
    return True










#------------------------------------------------------------------------------------
# Function: Process PDF
#------------------------------------------------------------------------------------

def extract_and_save_images_from_pdf(
        pdf_path: str,
        images_folder: Optional[str] = 'extracted_images',
        verbose: bool = True
        ) -> int :
    """Load PDF and extract all available images from it. Must 

    Args:
        pdf_path (Optional[str], optional): Path of PDF. Defaults to None.

    Returns:
        int: returns total number of images that had been stored
    """

        
    images_count = 0                                        # creating instance to store how much images saved yet
    doc = fitz.open(pdf_path)                               # loading pdf
    if len(doc) == 1:                                       # condition if pdf contain single page
        _pdf_page_to_png(doc[0])
        return True
    
    _clear_images_folder(images_folder)                     # deleting all pre-existing files
    
    for page_num, page in enumerate(doc):                   # iterating each page
        page_num += 1
        images = page.get_images(full=True)                 # extracting image(s) in current page

        for img_index, img in enumerate(images):            # iterating each image if more than one in same page
            xref = img[0]
            base_image = doc.extract_image(xref)
            doc_id = f"doc_{page_num}_{img_index}"          # generating unique doc_id


            # Convert raw image data to PIL Image
            img_data = base_image["image"]
            current_pil_img = Image.open(io.BytesIO(img_data)).convert('RGB')
            for image in os.listdir(images_folder):
                iterative_image_path = os.path.join(images_folder,image)
                if _are_images_identical(current_pil_img, iterative_image_path):
                    break
            else:
                with open(f"{images_folder}/{doc_id}.png", "wb") as f:   # loading current image (binary data)
                    f.write(base_image["image"])                # saving current image
                    images_count += 1
                    if verbose:
                        print(f"Image extracted and saved to directory (.../{images_folder}/{doc_id}).")



    return images_count








if __name__ == "__main__":

    pdf_path = "./content/Projection of Points.pdf"
    extract_and_save_images_from_pdf(pdf_path)

    # print(are_images_identical('extracted_images/doc_1_0.png', 'extracted_images/doc_7_2.png'))





