import PyPDF2
import docx


def load_pdf(filename):
    # Open the PDF file
    with open(filename, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfFileReader(file)

        # Create an empty string to hold the document text
        doc_text = ""

        # Loop over each page in the PDF file
        for page_num in range(pdf_reader.getNumPages()):
            # Get the page object
            page = pdf_reader.getPage(page_num)

            # Extract the text from the page and add it to the document text string
            doc_text += page.extractText()

        # Return the document text string
        return doc_text


def load_docx(filename):
    # Open the Word document
    doc = docx.Document(filename)

    # Create an empty string to hold the document text
    doc_text = ""

    # Loop over each paragraph in the document
    for para in doc.paragraphs:
        # Add the text from the paragraph to the document text string
        doc_text += para.text

    # Return the document text string
    return doc_text