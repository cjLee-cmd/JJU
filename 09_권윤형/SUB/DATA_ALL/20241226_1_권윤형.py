import os

os.environ['OPENAI_API_KEY'] = 'API_KEY'

#키 유효성 검사
print(f"[API KEY]\n{os.environ['OPENAI_API_KEY'][:-15]}" + "*" * 15)