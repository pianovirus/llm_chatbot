import os
import re
import unicodedata

import fitz  # PyMuPDF
import easyocr


class PDFToTextConverter:
    """
    ============================================================
    PDFToTextConverter
    ------------------------------------------------------------
    PDF 파일을 읽어서 텍스트(.txt) 파일로 변환하는 클래스입니다.

    이 클래스는 두 가지 소스에서 텍스트를 수집합니다.

    1) PDF 자체의 텍스트 레이어
       - 문서 본문, 복사 가능한 텍스트
       - 가장 정확하고 우선적으로 사용되는 소스

    2) PDF 페이지 안에 포함된 이미지의 OCR 결과
       - 스크린샷, 캡처 이미지, UI 이미지 안의 텍스트
       - 예: 버튼명, 메뉴명, 테이블 일부 텍스트 등

    최종 결과:
    - 각 PDF마다 하나의 .txt 파일 생성
    - 페이지 단위 구분선을 넣어 저장
    - 이미지 OCR 텍스트는 '[이미지 내 텍스트]' 태그와 함께 기록

    사용 목적:
    - PDF를 RAG 인덱싱용 텍스트 데이터로 변환
    - 이미지 속 텍스트까지 최대한 보존하여 검색 품질 향상
    ============================================================
    """

    def __init__(self):
        """
        ========================================================
        생성자
        --------------------------------------------------------
        OCR 엔진(EasyOCR)을 1회 로딩합니다.

        설정:
        - 한국어(ko), 영어(en) 동시 인식

        왜 생성자에서 로드하나?
        - EasyOCR 모델은 로딩 비용이 큼
        - 파일마다 매번 새로 로드하면 매우 느려짐
        - 한 번만 로드하고 전체 PDF 처리에 재사용하기 위함

        참고:
        - 이전에 이미지 요약(BLIP 등)을 사용할 수도 있었지만,
          현재는 OCR만 사용하여 이미지 속 글자만 정확히 추출하는 방향
        ========================================================
        """
        print("🚀 [Step 1] 분석 모델 로딩 중 (EasyOCR)...")
        self.reader = easyocr.Reader(["ko", "en"])
        print("✅ 분석 모델 로드 완료")

    def _clean_text(self, text):
        """
        ========================================================
        텍스트 정제 함수
        --------------------------------------------------------
        PDF에서 추출한 원시 텍스트를 정리합니다.

        처리 내용:
        1) None 또는 빈 문자열 방어
        2) Unicode NFC 정규화
           - macOS 환경에서 한글 자소 분리 문제 완화
        3) 연속된 공백/줄바꿈을 하나의 공백으로 정리
        4) 너무 짧은 텍스트 제거

        입력:
        - text: PDF 본문 추출 결과 문자열

        반환:
        - 정제된 문자열
        - 의미 없는 짧은 문자열이면 빈 문자열 반환

        왜 필요한가?
        - PDF 텍스트는 줄바꿈, 공백, 깨진 유니코드가 섞여 있을 수 있음
        - 후속 RAG 인덱싱 품질을 위해 기본 정제가 필요함
        ========================================================
        """
        if not text:
            return ""

        # macOS 환경에서 한글 자소 분리 방지용 정규화
        text = unicodedata.normalize("NFC", text)

        # 연속 공백/줄바꿈을 하나의 공백으로 정리
        text = re.sub(r"\s+", " ", text)

        # 너무 짧은 텍스트는 의미 없는 노이즈일 가능성이 높아 제거
        cleaned = text.strip()
        return cleaned if len(cleaned) >= 5 else ""

    def _analyze_image_with_ocr(self, image_bytes):
        """
        ========================================================
        이미지 OCR 분석 함수
        --------------------------------------------------------
        PDF 내 이미지 바이트 데이터를 받아 OCR을 수행합니다.

        입력:
        - image_bytes: PDF에서 추출한 이미지의 raw bytes

        처리 내용:
        1) EasyOCR로 이미지 안 텍스트 인식
        2) 결과를 하나의 문자열로 합침
        3) 결과가 있을 경우 '[이미지 내 텍스트]' 태그를 붙여 반환

        반환:
        - OCR 결과 문자열
        - 텍스트가 없거나 에러가 나면 빈 문자열 반환

        왜 필요한가?
        - PDF 내 스크린샷/UI 캡처 이미지에 중요한 텍스트가 있을 수 있음
        - 텍스트 레이어만으로는 잡히지 않는 내용을 보완하기 위함
        ========================================================
        """
        try:
            ocr_results = self.reader.readtext(image_bytes, detail=0)
            ocr_text = " ".join(ocr_results).strip()

            # OCR 결과가 있을 때만 태그를 붙여 반환
            return f"\n[이미지 내 텍스트]: {ocr_text}\n" if ocr_text else ""

        except Exception as e:
            print(f"⚠️ OCR 오류 발생: {e}")
            return ""

    def convert(self, file_path):
        """
        ========================================================
        단일 PDF 변환 함수
        --------------------------------------------------------
        하나의 PDF 파일을 읽어 텍스트 파일로 변환합니다.

        입력:
        - file_path: 변환할 PDF 파일 경로

        처리 절차:
        1) PDF 열기
        2) 각 페이지 순회
        3) 페이지 텍스트 레이어 추출 후 정제
        4) 페이지 내 이미지들 OCR 수행
        5) 페이지별 결과를 누적
        6) 최종적으로 converted_texts/<파일명>.txt 로 저장

        출력 파일 형식:
        - 페이지마다 --- Page N --- 구분선 삽입
        - 본문 텍스트 + OCR 텍스트 함께 저장

        예외 처리:
        - 특정 이미지 OCR 실패 시 전체 변환은 계속 진행
        - 개별 이미지 오류는 무시하고 다음 이미지로 이동
        ========================================================
        """
        doc = fitz.open(file_path)
        file_name = os.path.basename(file_path)
        output_data = []

        print(f"📖 {file_name} 변환 시작 (총 {len(doc)} 페이지)...")

        for page_num, page in enumerate(doc):
            # ----------------------------------------------------
            # 1) PDF의 텍스트 레이어 추출
            # ----------------------------------------------------
            # "text" 모드는 사람이 읽는 본문 문자열을 최대한 자연스럽게 추출
            content = page.get_text("text")
            cleaned = self._clean_text(content)

            # 페이지 구분선 + 본문 텍스트
            page_entry = f"--- Page {page_num + 1} ---\n{cleaned}\n"

            # ----------------------------------------------------
            # 2) 페이지 안 이미지 목록 추출 후 OCR 수행
            # ----------------------------------------------------
            # PDF 페이지에 포함된 이미지(xref 기반)를 순회하며
            # 각 이미지에서 OCR 텍스트를 추출
            image_list = page.get_images(full=True)

            for img_info in image_list:
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    ocr_content = self._analyze_image_with_ocr(image_bytes)
                    if ocr_content:
                        page_entry += ocr_content

                except Exception:
                    # 특정 이미지 처리 실패는 전체 변환을 막지 않음
                    continue

            output_data.append(page_entry)

        # --------------------------------------------------------
        # 3) 결과 저장
        # --------------------------------------------------------
        # 출력 폴더가 없으면 생성
        output_dir = "converted_texts"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 원본 파일명 뒤에 .txt를 붙여 저장
        # 예: manual.pdf -> manual.pdf.txt
        output_path = os.path.join(output_dir, f"{file_name}.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_data))

        doc.close()
        print(f"✅ 변환 완료: {output_path}")

    def convert_all_pdfs(self, source_dir):
        """
        ========================================================
        폴더 단위 일괄 PDF 변환 함수
        --------------------------------------------------------
        지정한 폴더 안의 모든 PDF 파일을 찾아 순차적으로 변환합니다.

        입력:
        - source_dir: PDF 파일들이 들어 있는 폴더 경로

        처리 절차:
        1) 폴더 존재 여부 확인
        2) .pdf 파일 목록 수집
        3) PDF가 없으면 안내 메시지 출력
        4) 각 PDF에 대해 convert() 호출

        사용 이유:
        - main()에서 반복 처리 로직을 깔끔하게 분리하기 위함
        - 단일 파일 처리와 전체 폴더 처리를 역할별로 나누기 위함
        ========================================================
        """
        if not os.path.exists(source_dir):
            print(f"⚠️ '{source_dir}' 폴더를 찾을 수 없습니다. 폴더를 생성하고 PDF를 넣어주세요.")
            return

        pdf_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".pdf")]

        if not pdf_files:
            print(f"📁 '{source_dir}' 폴더에 PDF 파일이 없습니다.")
            return

        for file_name in pdf_files:
            file_path = os.path.join(source_dir, file_name)
            self.convert(file_path)

        print("\n✨ 모든 PDF 변환 작업이 끝났습니다. 'converted_texts' 폴더를 확인하세요!")


def main():
    """
    ============================================================
    프로그램 진입점
    ------------------------------------------------------------
    PDF -> 텍스트 변환 전체 흐름을 시작합니다.

    처리 절차:
    1) PDFToTextConverter 객체 생성
    2) 소스 PDF 폴더 경로 설정
    3) 폴더 안의 모든 PDF를 일괄 변환

    현재 기본 입력 폴더:
    - ./sources

    기대 사용 방식:
    - sources 폴더에 PDF 넣기
    - 스크립트 실행
    - converted_texts 폴더에서 결과 확인
    ============================================================
    """
    converter = PDFToTextConverter()

    # PDF 파일이 들어 있는 입력 폴더
    source_dir = "./sources"

    converter.convert_all_pdfs(source_dir)


if __name__ == "__main__":
    main()