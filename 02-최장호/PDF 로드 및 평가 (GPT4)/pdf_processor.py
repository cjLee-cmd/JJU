from pdf_loader import PDFLoader
from pdf_evaluator import PDFEvaluator

class PDFProcessor:
    def __init__(self, file_path: str):
        """
        PDF 파일 로드 및 평가를 위한 클래스
        :param file_path: 평가할 PDF 파일 경로
        """
        self.file_path = file_path
        self.loader = PDFLoader(self.file_path)
        self.evaluator = PDFEvaluator()

    def run(self):
        """
        PDF 로드, 텍스트 추출, 평가 수행 및 결과 출력
        """
        # 텍스트 추출
        extracted_texts = self.loader.extract_texts()
        
        # GPT 모델로 평가
        evaluation_result = self.evaluator.evaluate(extracted_texts)
        
        # 결과 출력
        print("Evaluation for all loaders:\n")
        print(evaluation_result)
