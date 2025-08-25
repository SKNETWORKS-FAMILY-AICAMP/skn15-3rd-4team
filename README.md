# skn15-3rd-4team

# 1. 팀 소개

## 📑 Team 사고(思考)

|이름|직책|-|
|------|---|---|
|최민석 @Minsuk1014 |팀장, DB 관리, 프로젝트 총괄|<img width="102" height="116" alt="image" src="https://github.com/user-attachments/assets/0f7ec347-9f05-4747-878b-ae4db82ad4fa" /> |
|김민규 @kmklifegk |문서 기반 챗봇 프롬프트 튜닝 및 구현, 검수|<img width="102" height="116" alt="image" src="https://github.com/user-attachments/assets/4dac5983-f9d7-4bcf-bf9f-56aca4445042" />|
|김주형 @wugud09|Anki와 챗봇 연동 구현|<img width="102" height="116" alt="image" src="https://github.com/user-attachments/assets/4dac5983-f9d7-4bcf-bf9f-56aca4445042" /> |
|강민정 @kmj212936|검색 기반 챗봇 구현, Git 관리|<img width="102" height="116" alt="image" src="https://github.com/user-attachments/assets/d37f032d-a391-4ee7-a640-a42411291697" />|
|이세진 @isjini|검색 기반 챗봇 구현, Git 관리|<img width="102" height="116" alt="image" src="https://github.com/user-attachments/assets/e6c2a8d2-e5ab-4d14-b74e-220eb5cbb098" /> |
|최서린 @seorinchoi|문서 기반 챗봇 기본 기능 구현, 발표|<img width="102" height="116" alt="image" src="https://github.com/user-attachments/assets/f8f8083b-8b6a-49c6-9488-4e80d3bce37f" />|
















# 2. 프로젝트 기간
2025.08.22, 2025.08.25 (총 2일)

# 3. 프로젝트 개요


## 📕 프로젝트명
### Anki 기반 개인화 복습 챗봇

## ✅ 프로젝트 배경 및 목적



<img width="791" height="452" alt="image" src="https://github.com/user-attachments/assets/c208c1f9-151a-4699-8d71-41c074fde56e" />

---

지속적인 챗봇 기술의 발전으로 **ChatGPT**의 사용자는 기하급수적으로 증가하고 있다.  
특히 학습에 ChatGPT를 활용하는 사례가 크게 늘고 있으며, 실제로 2025년 7월 OpenAI에서는 학생들을 위해 **학습 전용 ChatGPT**를 공개하기도 했다.  

많은 학습자들은 새로운 개념을 이해하거나, 과제를 해결하거나, 자신만의 학습 플랜을 세울 때 GPT를 적극적으로 활용한다.  
궁금한 점이 있으면 GPT에게 질문하고, 필요한 답을 얻으며 학습 효율을 높일 수 있다.

그러나 기존 GPT만으로 학습을 진행하는 데에는 몇 가지 한계점이 존재한다.  


<img width="688" height="313" alt="image" src="https://github.com/user-attachments/assets/cec7611e-597d-49e6-befc-85859c1fa65d" />

---
* 먼저 **범용적인 학습만을 지원**하기 때문에 **사용자가 특정한 내용을 학습하고 싶을 때 사용하기엔 부적절**하다는 점이 있다. 이는 RAG를 활용하여 사용자가 원하는 학습 데이터를 사용할 수 있도록 개인화시키고자 한다.

* 또 10일, 20일이 지나면 대화가 누적되어 이전 내역을 찾기 어려워지고, 때문에 **장기적인 반복 학습이나 복습에는 다소 취약**하다.


#### 이러한 점을 개선하기 위해, 팀 사고에서는 학습용 웹 어플리케이션인 Anki와 연동하여 망각 주기에 따른 학습이 가능한 챗봇을 구현하고자 하였다.

  <img width="600" height="529" alt="image" src="https://github.com/user-attachments/assets/ccf7ad95-49d0-40cc-9e71-2a14028be531" />
  
---
* Anki는 위와 같은 망각 주기를 이용하여 사용자가 효과적으로 반복 학습을 할 수 있는 학습용 웹 어플리케이션이다.

* 챗봇은 Anki의 애드온인 Anki Connection을 연동하여, 사용자가 질문한 내용이나 학습한 내용을 바탕으로 **Anki에 저장**하고자 한다.

* 또, 사용자가 질문했을 때 시스템**DB, PDF 문서, 이미지 검색을 동시에 활용**하여 **최적의 답변**을 구현하여 환각 및 오정보를 최소한으로 한 학습 어플리케이션이 되도록 할 예정이다.


## 🔑 핵심 기능

- **다중 소스 RAG**  
  데이터베이스, 문서, 이미지 검색을 동시에 통합하여 더욱 풍부하고 정확한 답변 제공

- **양방향 처리**  
  - 사용자에게는 실시간으로 답변 제공  
  - 동시에 해당 답변을 DB에 최적화된 형태로 저장 → 학습용 + 검색용으로 활용

- **지속적 복습 지원**  
  DB에 저장된 질문/답변을 MCP + Anki API를 통해 다시 불러와 **개인화된 복습** 가능

- **PostgreSQL 기반 구조화 저장**  



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

### 개발 환경
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

### 소통
![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white)


# 5. 수행결과
### 모델 그래프
<img width="548" height="389" alt="image" src="https://github.com/user-attachments/assets/81d4958e-b0a9-4b46-a34e-8f7f8f46e255" />

### 시연 영상

[![시연 영상](https://img.youtube.com/vi/UJvSXeK98tw/maxresdefault.jpg)](https://youtu.be/UJvSXeK98tw)

### 라우팅

|1|2|
|------|---|
|<img width="669" height="579" alt="image" src="https://github.com/user-attachments/assets/ae5874d6-0ce8-41f7-8c1a-2dbf70240204" />|<img width="669" height="579" alt="image" src="https://github.com/user-attachments/assets/b823d867-a577-4627-b131-1bd3f9e04aa3" />|


## Anki 저장 목록과 대화 내역 비교
<img width="678" height="505" alt="image" src="https://github.com/user-attachments/assets/f549e44a-615d-4e4b-aa15-64a748052d2c" />



# 6. 다음 프로젝트 목표...
<img width="1910" height="956" alt="image" src="https://github.com/user-attachments/assets/8e1791b9-be39-4c5f-9a87-cca46b11624a" />


[검색기능을 보완하여 구글 검색 API를 이용한 이미지 검색을 완성시키기]

:현재 프로젝트의 검색 기능을 확장하여, 단순 텍스트 검색을 넘어 
**이미지 기반 검색**과 **생성형 AI를 활용한 맞춤 이미지 제공**을 목표로 함.  

구체적으로는 다음과 같은 내용을 계획한다:

1. **이미지 기반 검색 (Image-to-Image Retrieval)**  
   - 사용자가 이미지를 업로드하면 해당 이미지와 유사한 자료를 검색하여 추천  
   - 시각적 비교가 중요한 영역에서 활용 가능

2. **프롬프트 기반 이미지 변환 (Prompt-guided Editing)**  
   - 사용자가 조건·목적·컬러 변경 요청을 입력하거나 클릭  
   - 해당 조건에 맞추어 기존 이미지를 수정/재생성  
   - 예: 특정 색상 변경, 분위기 조정, 특정 요소 강조 등

3. **멀티모달 생성 (Text/Place → Image/3D)**  
   - 사용자가 텍스트 키워드나 장소 정보를 입력하면 이를 기반으로 이미지 생성  
   - 더 나아가 3D 시각화로 확장하여 제공하는 기능도 실험

4. **검색 + 생성 융합**  
   - 검색된 결과를 단순히 보여주는 것을 넘어,  
   - 생성형 AI와 결합해 검색 결과를 사용자 맞춤형으로 재해석·재창조할 수 있는 기능 구현

👉 이를 통해 단순한 검색 기능을 넘어서,  
**사용자가 직관적으로 원하는 이미지를 검색·수정·생성할 수 있는 사용자 맞춤 이미지 검색 시스템**으로 발전시키고자 함.


# 7. 한 줄 회고

|이름|회고|
|------|---|
|최서린 |기존에 배웠던 내용들을 합쳐서 새로운 결과물이 나올 수 있다는 게 신기했습니다. 다들 자기 자리에서 열심히 준비해주신 덕분에 난이도 있는 프로젝트인데도 금방 완성할 수 있었던 것 같습니다!|
|이세진|웹 기반 검색 챗봇을 구현하면서 tavily와 Google API를 활용한 이미지/문서 검색 기능을 다루고,  이미지 검색을 불러오는 방식을 고민하면서 API 활용의 한계와 확장 가능성을 체감 할 수있었습니다.단순 검색을 넘어서 사용자 경험을 풍부하게 만드는 방향성을 팀원들과 함께 고민할 수 있었던 점이 의미있는 시간이었습니다.|
|강민정 |수업 시간에 배운 내용을 직접 활용해 Anki MCP 기반 개인화 복습 챗봇을 구현해본 경험이 신기하고 뜻깊었습니다. 단순히 따라하기에 그치지 않고, 더 나은 플로우를 고민하며 의견을 제안했던 과정도 프로젝트를 진행하는 데 의미 있는 시간이었습니다. 또한 이번 챗봇 개발을 바탕으로, 다음 프로젝트 역할에서는 이미지 검색 기능을 심화적으로 다뤄보고싶다는 흥미와 동기부여도 생겼습니다. 프로젝트를 진행하면서 어려운 부분이나 고민되는 지점이 있을 때, 팀원들과 함께 의견을 나누며 협력했던 순간들 역시 소중한 경험으로 남았습니다.|
|김민규 |훌륭한 조장과 팀원을 만난 덕분에 자신이 맡은 부분 뿐 아니라 다른 조원까지 도와줘서 감사한 마음이 드는 프로젝트였습니다.
|최민석 |다들 열심히 각자의 자리에서 열심히 해줘서 어려움 없이 진행했던 것 같습니다. 또한 랭그래프가 생각보다 편한 라이브러리같다는 생각이 들었습니다.
|김주형 |모든 팀원이 각자의 기능을 최선을 다해 구현했고, 꾸준한 의사소통과 협업을 통해 프로젝트를 완성할 수 있었습니다. 지금까지 배운 내용을 실제로 활용해 볼 수 있었으며, 팀원들과의 협업 과정에서 많은 것을 배우고 성장할 수 있었던 값진 경험이었습니다.





