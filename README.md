# skn15-3rd-4team

# 1. 팀 소개
<img width="172" height="152" alt="image" src="https://github.com/user-attachments/assets/0f7ec347-9f05-4747-878b-ae4db82ad4fa" />최민석 @Minsuk1014  
<img width="102" height="116" alt="image" src="https://github.com/user-attachments/assets/4dac5983-f9d7-4bcf-bf9f-56aca4445042" />김민규 @kmklifegk 
<img width="102" height="116" alt="image" src="https://github.com/user-attachments/assets/4dac5983-f9d7-4bcf-bf9f-56aca4445042" /> 김주형 @wugud09



<img width="102" height="116" alt="image" src="https://github.com/user-attachments/assets/d37f032d-a391-4ee7-a640-a42411291697" />강민정 @kmj212936
<img width="102" height="116" alt="image" src="https://github.com/user-attachments/assets/e6c2a8d2-e5ab-4d14-b74e-220eb5cbb098" /> 이세진 @isjini
<img width="102" height="116" alt="image" src="https://github.com/user-attachments/assets/f8f8083b-8b6a-49c6-9488-4e80d3bce37f" />최서린 @seorinchoi








# 2. 프로젝트 기간
2025.08.22 ~ 2025.08.25 (총 2일)

# 3. 프로젝트 개요


## 📕 프로젝트명
Anki MCP 기반 개인화 복습 챗봇

## ✅ 프로젝트 배경 및 목적

챗지피티는 사용자의 학습 지도 역할도 수행하고 있음. OpenAI는 최근 공부하는 학생들을 위한 ‘학습 모드’를 내놓았음.

https://www.betanews.net/article/view/beta202507300054

하지만 이러한 학습 모드는 **범용적인 학습만을 지원**하기 때문에 **사용자가 특정 시험을 준비하고 싶을 때 사용하기엔 부적절**함

또 **반복 학습이나 복습에는 다소 취약**한 부분이 있음

이러한 부분을 보완하기 위해 **학습용 웹 어플리케이션인 ANKI와 연동할 수 있는 챗봇**을 구현함.

챗봇은 사용자가 질문했을 때 시스템**DB, PDF 문서, 이미지 검색을 동시에 활용**하여 **최적의 답변**을 구현하는 기능을 가지고 있음.






## ❤️ 기대효과

### 👌 학습용(PostgreSQL, Anki API 연동)


✅ 질문·답변을 4지선다 문제와 해설로 변환하여 **복습 가능**하게 저장


✅ DB에 입력한 학습용 자료 뿐만 아니라 웹에 올라와 있는 다양한 온라인 자료들을 활용하여 사용자에게 **폭 넓은 학습을 지원**





### 👌검색 최적화용(PostgreSQL)


✅ **질문·답변을 요약하고, “내가 했던 질문인지”를 추적할 수 있도록 저장**


이를 통해 **단순 Q/A 챗봇을 넘어, 개인화된 복습이 가능한 지식 관리형 GPT 시스템을 구현**



## 👉 이 프로젝트의 주요 사용자 
🏫학습자(학생, 자격증 준비생, 자기계발러)

👩‍🏫교육자(교사, 강사), 직장인(사내 학습자, 지식 관리자)

🥸일반 학습자(언어, 취미 학습자)로 나눌 수 있음 

**모두 공통적으로 “질문-답변을 단순 소비로 끝내지 않고, 나중에 복습/재활용”이 필요한 사람들이 주 타겟.**


# 4. 기술 스택
### 기술 스택

![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)

### 소통

![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white)


# 5. 수행결과



# 6. 한 줄 회고

|이름|회고|
|------|---|
|최서린 |기존에 배웠던 내용들을 합쳐서 새로운 결과물이 나올 수 있다는 게 신기했습니다. 다들 자기 자리에서 열심히 준비해주신 덕분에 난이도 있는 프로젝트인데도 금방 완성할 수 있었던 것 같습니다!|
|이세진|웹 기반 검색 챗봇을 구현하면서 tavily와 Google API를 활용한 이미지/문서 검색 기능을 다루고,  이미지 검색을 불러오는 방식을 고민하면서 API 활용의 한계와 확장 가능성을 체감 할 수있었습니다.단순 검색을 넘어서 사용자 경험을 풍부하게 만드는 방향성을 팀원들과 함께 고민할 수 있었던 점이 의미있는 시간이었습니다.|
|강민정 |수업 시간에 배운 내용을 직접 활용해 Anki MCP 기반 개인화 복습 챗봇을 구현해본 경험이 신기하고 뜻깊었습니다. 단순히 따라하기에 그치지 않고, 더 나은 프레임워크를 만들기 위해 고민하며 의견을 제안했던 과정도 프로젝트를 진행하는 데 의미 있는 시간이었습니다. 또한 이번 챗봇 개발을 바탕으로, 다음 프로젝트 역할에서는 DB 영역 또한 진행해 보고싶다는 흥미와 동기부여도 생겼습니다. 프로젝트를 진행하면서 어려운 부분이나 고민되는 지점이 있을 때, 팀원들과 함께 의견을 나누며 협력했던 순간들 역시 소중한 경험으로 남았습니다.|
|김민규 |훌륭한 조장과 팀원을 만난 덕분에 자신이 맡은 부분 뿐 아니라 다른 조원까지 도와줘서 감사한 마음이 드는 프로젝트였다.

최민석:
김주형:





