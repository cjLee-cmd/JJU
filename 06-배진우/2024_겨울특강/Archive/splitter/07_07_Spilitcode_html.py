from langchain_text_splitters import HTMLHeaderTextSplitter

html_string = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>자기소개서</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            padding: 0;
            background-color: #f9f9f9;
        }
        h1, h2, h3 {
            color: #003366;
        }
        h1 {
            text-align: center;
            margin-bottom: 40px;
        }
        section {
            background-color: #ffffff;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        ul {
            margin-left: 20px;
        }
        strong {
            color: #0055a5;
        }
    </style>
</head>
<body>
    <h1>자기소개서</h1>
    
    <section>
        <h2>자기소개</h2>
        <p>저는 중학교 2학년 무렵부터 스스로 흥미를 느낀 코딩에 도전하며, <strong>‘구상한 아이디어를 현실화하는 것’</strong>에 몰입해 왔습니다.</p>
        <ul>
            <li><strong>2019년</strong>: Win32 API를 활용하여 팩맨 게임을 구현하며 구조적 프로그래밍과 그래픽 처리의 기초를 다졌습니다.</li>
            <li><strong>2021년</strong>: Unity를 활용해 2D 기반의 2인 플레이 게임을 제작하며 게임 로직 설계의 감각을 키울 수 있었습니다.</li>
        </ul>
        <p>이러한 경험을 통해 저는 단순히 코드를 짜는 것에 그치지 않고, <strong>끊임없이 새로운 기능과 개선점을 모색하는 ‘문제 해결형’ 개발자</strong>로 성장했습니다.</p>
        <p>이후 인공지능 분야에 매력을 느껴 인공지능 전공으로 대학에 진학하였고, <strong>골절 진단 시스템</strong>이나 <strong>육계 무게 예측 모델</strong> 등 다양한 프로젝트에 참여하면서 인공지능 기술을 실용적이며 가치 있는 결과물로 구현하는 경험을 쌓고 있습니다.</p>
        <p>이러한 과정에서 저는 주어진 문제를 구조적으로 접근하고, 알고리즘과 모델을 탐색하여 최적의 솔루션을 도출하는 능력을 발전시켜 왔습니다.</p>
        <p>나아가 팀원과의 의사소통을 통해 다양한 관점을 조정하고, 협력적인 환경에서 성장할 수 있는 역량 역시 갖추게 되었습니다.</p>
        <p>결국 제가 지닌 적성은 <strong>‘아이디어를 논리적으로 구현하는 능력’</strong>과 <strong>‘새로운 기술을 유연하게 흡수하는 학습 역량’</strong>에 기반하고 있습니다.</p>
    </section>
    
    <section>
        <h2>자신의 장단점</h2>
        
        <h3>장점</h3>
        <p>저는 한 번 관심을 두게 된 주제나 과제에 대해 <strong>깊이 파고들며, 끝까지 해결책을 찾기 위해 몰두하는 집요함</strong>을 강점으로 가지고 있습니다.</p>
        <p>이러한 집중력 덕분에 문제 해결 과정에서 남들이 놓치기 쉬운 세부사항까지 꼼꼼히 점검하고, 결국 원하는 목표에 도달하는 경험을 자주 했습니다.</p>
        
        <h3>단점</h3>
        <p>제 단점은 <strong>흥미를 잃은 영역에 대해서는 집중력이 급격히 떨어진다는 점</strong>입니다.</p>
        <p>이로 인해 때때로 중요한 업무를 균형 있게 진행하기 어렵거나, 팀 내 협업 과정에서 제 역할을 충분히 못 하는 상황이 발생할 수 있습니다.</p>
        <p>이런 단점을 개선하기 위해서는 업무 전체 흐름을 파악하고, 관심을 덜 느끼는 부분이라도 목표 달성을 위해 필요한 과정임을 인식하며 <strong>자기 동기부여 방식을 다양화</strong>하고 있습니다.</p>
        <p>이런 노력을 통해 더욱 <strong>지속적이고 균형 잡힌 성과</strong>를 내고자 합니다.</p>
    </section>
    
    <section>
        <h2>교육 수료 후 계획 (구체적으로)</h2>
        <p>저는 <strong>2024년 2월경 군입대</strong>를 예정하고 있으며, 정확한 입대일은 아직 확정되지 않았습니다.</p>
        <p>입대 후 자대배치를 받고 생활에 적응한 뒤, 군 복무 기간 동안 <strong>꾸준히 학습을 이어나갈 계획</strong>입니다.</p>
        
        <h3>1차 목표</h3>
        <ul>
            <li><strong>『밑바닥부터 시작하는 딥러닝』 시리즈(1~5권) 완독</strong>
                <ul>
                    <li>딥러닝 전반에 대한 이론적 기반과 실무적 감각을 다질 것입니다.</li>
                </ul>
            </li>
        </ul>
        
        <h3>추가 학습 목표</h3>
        <ul>
            <li><strong>선형대수학, 확률·통계 등 수학적 기초 역량 강화</strong></li>
            <li><strong>영어 독해 및 기술 문서 이해 능력 향상</strong>
                <ul>
                    <li>전문 지식 습득에 필요한 토대를 더욱 탄탄히 할 것입니다.</li>
                </ul>
            </li>
        </ul>
        
        <h3>장기 목표</h3>
        <ul>
            <li><strong>군 복무 후 복학 시점</strong>:
                <ul>
                    <li>보다 깊이 있는 연구 활동에 매진할 수 있는 역량을 갖추게 될 것입니다.</li>
                </ul>
            </li>
            <li><strong>학업 이어가기 (4학년 즈음)</strong>:
                <ul>
                    <li>인공지능 관련 대학원 진학을 적극적으로 고려할 계획입니다.</li>
                </ul>
            </li>
            <li><strong>최종 목표</strong>:
                <ul>
                    <li>인공지능 분야에서 새로운 패러다임을 제시할 수 있는 모델을 개발하고, 이를 통해 <strong>학문적·산업적 흐름을 주도하는 연구자</strong>로 성장하고자 합니다.</li>
                </ul>
            </li>
        </ul>
        
        <p>이러한 <strong>장기적 목표</strong>에 한 걸음씩 다가가기 위해 지금부터 <strong>체계적인 학습과 자기계발</strong>에 힘쓸 것입니다.</p>
    </section>
</body>
</html>
"""

headers_to_split_on = [
    ("h1", "Header 1"),  # 분할할 헤더 태그와 해당 헤더의 이름을 지정합니다.
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

# 지정된 헤더를 기준으로 HTML 텍스트를 분할하는 HTMLHeaderTextSplitter 객체를 생성합니다.
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# HTML 문자열을 분할하여 결과를 html_header_splits 변수에 저장합니다.
html_header_splits = html_splitter.split_text(html_string)
# 분할된 결과를 출력합니다.
for header in html_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")