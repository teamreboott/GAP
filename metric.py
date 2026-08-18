import json 
import pprint

# JSON 파일 열기 - load가 아니라 open으로 파일을 먼저 열어야 함
with open("/home/work/hongjunchoi/git_repo/TFLOP/results/pubtabnet_experiment_reproduce/pubtabnet_experiment_reproduce/epoch_40_step_157850/full_model_inference_0_1.json", "r") as f:
    test_json = json.load(f)

# 첫 번째 항목만 예쁘게 출력

# print(list(test_json.keys()))
first_key = list(test_json.keys())[0]
print(f"첫 번째 이미지: {first_key}")
print("-" * 50)
pprint.pprint(test_json[first_key], indent=2)
# print("-" * 50)
# print(f"\n전체 이미지 수: {len(test_json)}")

# # 모든 키 확인
print(f"\n첫 번째 항목의 키들: {list(test_json[first_key].keys())}")