import PyPDF2
from config import LOCAL_PDF_PATH

class PDFProcessor:
    def __init__(self):
        self.text = self._extract_text_from_pdf()

    def _extract_text_from_pdf(self):
        """Extract text from the local PDF file"""
        try:
            with open(LOCAL_PDF_PATH, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error reading PDF: {str(e)}")
            return "" 