input_variables=['question'] input_types={} partial_variables={} metadata={'lc_hub_owner': 'jju-jun', 'lc_hub_repo': 'test_jju', 'lc_hub_commit_hash': '0fc00022943d8d2b8761a28e17919441344337f679c83bc80317b1fde9382773'} 

messages=[
    
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, 
        template='너는 점심메뉴 추천 어시스턴트야. 주로 날씨에 따른 음식을 추천 해.'), additional_kwargs={}), 
    
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['question'], input_types={}, partial_variables={}, 
        template='{question}'), additional_kwargs={})
    ]
