#utils.py
import re

def parse_anki_cards(text_blob: str) -> list[dict]:
    """
    LLM이 생성한 텍스트 덩어리에서
    Anki 카드(앞면/뒷면)들을 파싱하여 리스트로 반환합니다.
    4지선다 형식도 올바르게 처리합니다.
    """
    cards = []
    
    # 정규표현식 패턴: '앞면' 키워드 앞뒤의 공백(들여쓰기 포함)을 허용합니다.
    potential_cards = re.split(r"\s*앞면\s*", text_blob)

    # 첫 번째 분할 결과는 '앞면' 이전의 텍스트이므로 무시합니다.
    for card_text in potential_cards[1:]:
        # '뒷면' 키워드 앞뒤의 공백도 허용합니다.
        parts = re.split(r"\s*뒷면\s*", card_text)
        if len(parts) == 2:
            front = parts[0].strip()
            back = parts[1].strip()
            
            # 🔥 4지선다 문제 형식 개선
            # Question으로 시작하는 카드의 경우, 줄바꿈 형식을 보존
            if front.startswith("Question:"):
                # Question과 선택지들을 올바른 형식으로 정리
                front = format_multiple_choice_question(front)
            
            # 앞면과 뒷면 내용이 모두 존재할 경우에만 유효한 카드로 간주합니다.
            if front and back:
                cards.append({"front": front, "back": back})
    
    print(f"카드 파싱 완료: {len(cards)}개 카드 발견")
    return cards

def format_multiple_choice_question(question_text: str) -> str:
    """
    4지선다 문제를 올바른 형식으로 정리합니다.
    
    입력: "Question: 어쩌구 A. 선택지1 B. 선택지2 C. 선택지3 D. 선택지4"
    출력: "Question: 어쩌구\nA. 선택지1\nB. 선택지2\nC. 선택지3\nD. 선택지4"
    """
    # Question: 부분과 선택지들을 분리
    if "Question:" in question_text:
        # Question: 뒤의 내용을 가져옴
        content = question_text.split("Question:", 1)[1].strip()
        
        # A., B., C., D. 패턴으로 선택지들을 찾아서 줄바꿈으로 분리
        # 정규표현식으로 A., B., C., D. 앞에서 분할
        parts = re.split(r'\s*([A-D]\.)', content)
        
        if len(parts) > 1:  # 선택지가 있는 경우
            question_part = parts[0].strip()
            formatted_question = f"Question: {question_part}"
            
            # 선택지들 처리 (홀수 인덱스는 'A.', 'B.' 등이고, 짝수 인덱스는 내용)
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    option_letter = parts[i]  # 'A.', 'B.' 등
                    option_content = parts[i + 1].strip()
                    formatted_question += f"\n{option_letter} \n{option_content}"
            
            return formatted_question
    
    # 형식이 맞지 않으면 원본 그대로 반환
    return question_text