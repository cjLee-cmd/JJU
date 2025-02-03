UPSTAGE_API_KEY = 'up_TWQTG4XVSyCR2cPJw2dIE7181aG5w'

import os


os.environ["OPENAI_API_KEY"] = "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"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "test_JJU_YuSangMin"

from langchain_upstage import UpstageLayoutAnalysisLoader

file_path = '/workspaces/JJU-1/04-유상민/250107과제_2/SPRI_AI_Brief_2023년12월호_F.pdf'

loader = UpstageLayoutAnalysisLoader(
    file_path,
    output_type = 'html',
    split = 'page',
    use_ocr = True,
    exclude = ["header", 'footer'],
    api_key= UPSTAGE_API_KEY

)


docs1 = loader.load()

for doc in docs1[6]:
  print(doc)

loader = UpstageLayoutAnalysisLoader(
    file_path,
    output_type = 'html',
    split = 'element',
    use_ocr = True,
    exclude = ["header", 'footer'],
    api_key= UPSTAGE_API_KEY

)


docs2 = loader.load()

for doc in docs2[6]:
  print(doc)