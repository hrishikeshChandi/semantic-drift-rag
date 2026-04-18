from typing import List, Union
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFDirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document
from pathlib import Path


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        print(
            f"Initializing DocumentProcessor with chunk_size={chunk_size} and chunk_overlap={chunk_overlap}"
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def load_from_url(self, url: str) -> List[Document]:
        print(f"Loading document from URL: {url}")
        return WebBaseLoader(url).load()

    def load_from_pdf_directory(
        self, directory_path: Union[str, Path]
    ) -> List[Document]:
        print(f"Loading PDF documents from directory: {directory_path}")
        return PyPDFDirectoryLoader(directory_path).load()

    def load_from_pdf_file(self, file_path: Union[str, Path]) -> List[Document]:
        print(f"Loading PDF document from file: {file_path}")
        return PyPDFLoader(file_path).load()

    def load_from_text_file(self, file_path: Union[str, Path]) -> List[Document]:
        print(f"Loading text document from file: {file_path}")
        return TextLoader(file_path, encoding="utf-8").load()

    def load_documents(self, sources: List[str]) -> List[Document]:
        print(f"Loading documents from sources: {sources}")
        documents: List[Document] = []

        for src in sources:
            src_path = Path(src)

            if src.startswith("http://") or src.startswith("https://"):
                documents.extend(self.load_from_url(src))

            elif src_path.is_file():
                if src_path.suffix.lower() == ".txt":
                    documents.extend(self.load_from_text_file(str(src_path)))
                elif src_path.suffix.lower() == ".pdf":
                    documents.extend(self.load_from_pdf_file(str(src_path)))
                else:
                    raise ValueError(f"Unsupported file type: {src}")

            elif src_path.is_dir():
                print(f"Scanning directory: {src_path}")
                for file in src_path.iterdir():
                    if file.suffix.lower() == ".pdf":
                        documents.extend(self.load_from_pdf_file(str(file)))
                    elif file.suffix.lower() == ".txt":
                        documents.extend(self.load_from_text_file(str(file)))

            else:
                raise ValueError(f"Invalid source: {src}")

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        print(f"Splitting documents into chunks...")
        return self.text_splitter.split_documents(documents)

    def load_and_split_documents(self, sources: List[str]) -> List[Document]:
        documents = self.load_documents(sources)
        print(f"Loaded {len(documents)} documents.")
        return self.split_documents(documents)
