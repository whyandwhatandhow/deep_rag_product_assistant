import requests
import json

# 测试 /query 接口
url = "http://localhost:8002/api/v1/query"
headers = {"Content-Type": "application/json"}
data = {"question": "AnatoMask 是什么？"}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)  # 增加超时时间到 30 秒
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.json()}")
except Exception as e:
    print(f"请求失败: {e}")

# 测试 /health 接口
health_url = "http://localhost:8000/api/v1/health"
try:
    health_response = requests.get(health_url, timeout=10)
    print(f"\n健康检查状态码: {health_response.status_code}")
    print(f"健康检查响应: {health_response.json()}")
except Exception as e:
    print(f"健康检查失败: {e}")
