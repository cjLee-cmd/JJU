# RAG CHUNK Optimizer

### 240108
[MVP 구현]
- input으로 path 받기 -> path의 확장자 명을 따라서 loader로 파일 load -> 로드된 파일의 형식의 맞춰서 splitter를 이용해 split하기 -> 디버깅 pring문 만들기
- 프롬프트 엔지니어링 -> chat gpt 4o에게 위 결과를 반환해 점수와 이유를 물어보기

프롬프트
: 당신은 지금부터 llm 평가원 입니다.
    주어지는 결과를 보고 이에 대해서 평가 한 이후, 왜 그렇게 평가 했는지에 대해서 설명하시오.
    결과는 어떠한 파일을 langchain의 loader를 사용해 불러왔으며, 이를 langchain의 splitter를 사용해 나눈 것이다.
    결과는 여러 방법으로 load하고, 여러 방법으로 split한 것이다. 주어진 각각의 모든 방법을 평가 해라.
    이때 llm이 splitter한 파일을 잘 이해할 수 있는지에 대해서 평가를 하면 된다.
    평가 방법아래와 같다.
    input file type : 어떤 확장자 파일이 들어왔는지
    Loader : langchain의 어떤 라이브러리를 사용해서 load했는지
    splitter : langchain의 어떤 라이브러리를 사용해서 spilt했는지
    score : 평가 점수
    result : 이유(적절하고 직관적으로 설명)

    ex) {output}

    {results}

결과

평가 결과는 다음과 같습니다:

1. 
   - **input file type**: pdf
   - **Loader**: py_mu_pdf
   - **splitter**: charcter_text_splitter
   - **score**: 75
   - **result**: Loader가 PDF 파일을 잘 로드했으며, splitter가 텍스트를 적절히 나누어 LLM이 이해하기에 무리가 없다.

2. 
   - **input file type**: pdf
   - **Loader**: py_pdf_mu2loader
   - **splitter**: charcter_text_splitter
   - **score**: 75
   - **result**: Loader가 PDF 파일을 잘 로드했으며, splitter가 텍스트를 적절히 나누어 LLM이 이해하기에 무리가 없다.

3. 
   - **input file type**: pdf
   - **Loader**: pdf_minerloader
   - **splitter**: charcter_text_splitter
   - **score**: 60
   - **result**: Loader는 PDF 파일을 잘 로드했으나, splitter가 텍스트를 지나치게 세분화하여 문맥이 끊기는 부분이 있어 이해에 어려움이 있을 수 있다.

4. 
   - **input file type**: pdf
   - **Loader**: pdf_plumber
   - **splitter**: charcter_text_splitter
   - **score**: 75
   - **result**: Loader가 PDF 파일을 잘 로드했으며, splitter가 텍스트를 적절히 나누어 LLM이 이해하기에 무리가 없다.

5. 
   - **input file type**: pdf
   - **Loader**: py_mu_pdf
   - **splitter**: token_text_splitter
   - **score**: 70
   - **result**: Loader가 PDF 파일을 잘 로드했으며, splitter가 토큰 단위로 나누어 LLM이 이해하기에 적절하다.

6. 
   - **input file type**: pdf
   - **Loader**: py_pdf_mu2loader
   - **splitter**: token_text_splitter
   - **score**: 70
   - **result**: Loader가 PDF 파일을 잘 로드했으며, splitter가 토큰 단위로 나누어 LLM이 이해하기에 적절하다.

7. 
   - **input file type**: pdf
   - **Loader**: pdf_minerloader
   - **splitter**: token_text_splitter
   - **score**: 55
   - **result**: Loader는 PDF 파일을 잘 로드했으나, splitter가 텍스트를 지나치게 세분화하여 문맥이 끊기는 부분이 있어 이해에 어려움이 있을 수 있다.

8. 
   - **input file type**: pdf
   - **Loader**: pdf_plumber
   - **splitter**: token_text_splitter
   - **score**: 70
   - **result**: Loader가 PDF 파일을 잘 로드했으며, splitter가 토큰 단위로 나누어 LLM이 이해하기에 적절하다.

9. 
   - **input file type**: pdf
   - **Loader**: py_mu_pdf
   - **splitter**: recursive_character_text_splitter
   - **score**: 65
   - **result**: Loader가 PDF 파일을 잘 로드했으나, splitter가 텍스트를 지나치게 세분화하여 문맥이 끊기는 부분이 있어 이해에 어려움이 있을 수 있다.

10. 
    - **input file type**: pdf
    - **Loader**: py_pdf_mu2loader
    - **splitter**: recursive_character_text_splitter
    - **score**: 65
    - **result**: Loader가 PDF 파일을 잘 로드했으나, splitter가 텍스트를 지나치게 세분화하여 문맥이 끊기는 부분이 있어 이해에 어려움이 있을 수 있다.

11. 
    - **input file type**: pdf
    - **Loader**: pdf_minerloader
    - **splitter**: recursive_character_text_splitter
    - **score**: 50
    - **result**: Loader는 PDF 파일을 잘 로드했으나, splitter가 텍스트를 지나치게 세분화하여 문맥이 끊기는 부분이 있어 이해에 어려움이 있을 수 있다.

12. 
    - **input file type**: pdf
    - **Loader**: pdf_plumber
    - **splitter**: recursive_character_text_splitter
    - **score**: 65
    - **result**: Loader가 PDF 파일을 잘 로드했으나, splitter가 텍스트를 지나치게 세분화하여 문맥이 끊기는 부분이 있어 이해에 어려움이 있을 수 있다.

13. 
    - **input file type**: pdf
    - **Loader**: py_mu_pdf
    - **splitter**: semanticchunker
    - **score**: 80
    - **result**: Loader가 PDF 파일을 잘 로드했으며, splitter가 의미 단위로 나누어 LLM이 이해하기에 매우 적절하다.

14. 
    - **input file type**: pdf
    - **Loader**: py_pdf_mu2loader
    - **splitter**: semanticchunker
    - **score**: 80
    - **result**: Loader가 PDF 파일을 잘 로드했으며, splitter가 의미 단위로 나누어 LLM이 이해하기에 매우 적절하다.

15. 
    - **input file type**: pdf
    - **Loader**: pdf_minerloader
    - **splitter**: semanticchunker
    - **score**: 75
    - **result**: Loader가 PDF 파일을 잘 로드했으며, splitter가 의미 단위로 나누어 LLM이 이해하기에 적절하다.

16. 
    - **input file type**: pdf
    - **Loader**: pdf_plumber
    - **splitter**: semanticchunker
    - **score**: 80
    - **result**: Loader가 PDF 파일을 잘 로드했으며, splitter가 의미 단위로 나누어 LLM이 이해하기에 매우 적절하다.

각 방법은 PDF 파일을 로드하고 텍스트를 나누는 데 있어 다양한 수준의 적절성을 보여주었습니다. Semanticchunker를 사용한 경우가 가장 높은 점수를 받았으며, 이는 의미 단위로 텍스트를 나누어 LLM이 문맥을 더 잘 이해할 수 있도록 했기 때문입니다.