from file_loader import load_file

if __name__ == "__main__":
    # 테스트 파일 경로 입력
    test_file_path = "10_여명구/91_Practice_Project/02_LoaderSplitterPackage/data/SS0002_04_S19_1107_sound.csv"

    try:
        # 파일 로드
        results = load_file(test_file_path)

        # 결과 출력
        if isinstance(results, dict):  # PDF 처리 결과
            for loader_name, result in results.items():
                print(f"로더: {loader_name}")
                if isinstance(result, list):
                    print(f"로드된 문서 개수: {len(result)}")
                    print(f"첫 번째 문서 내용: {result[0]}")
                else:
                    print(f"오류 발생: {result}")
                print("-" * 40)
        else:  # 다른 파일 처리 결과
            print(f"로드된 문서 개수: {len(results)}")
            print("첫 번째 문서 내용:")
            print(results[0])
    except Exception as e:
        print(f"오류 발생: {e}")
