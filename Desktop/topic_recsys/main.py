import os
import requests
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

CATEGORIES = {
    "기술/AI": "technology",
    "경제/산업": "business",
    "정치/사회": "general",
    "과학/환경": "science"
}

# ---------------------------
# 뉴스 수집
# ---------------------------
def fetch_news(category):

    url = "https://newsapi.org/v2/top-headlines"

    params = {
        "apiKey": NEWS_API_KEY,
        "country": "us",
        "category": category,
        "pageSize": 10
    }

    response = requests.get(url, params=params)

    data = response.json()

    articles = []

    for a in data["articles"]:
        if a["title"]:
            articles.append(a["title"])

    return articles


# ---------------------------
# 토론 주제 생성
# ---------------------------
def generate_topics(category_ko, articles):

    news_text = "\n".join([f"- {a}" for a in articles[:8]])

    prompt = f"""
다음 뉴스 제목을 보고 토론 주제를 만들어라.

{news_text}

'{category_ko}' 분야에서 사람들이 토론할 만한
핫한 토론 논제를 2개 생성해라.

조건
- 찬반 토론 가능
- 한국어
- 사회적 논쟁 가능

JSON 형식으로 출력

{{
 "topics":[
  {{"topic":"..."}},
  {{"topic":"..."}}
 ]
}}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    text = response.output_text

    return json.loads(text)


# ---------------------------
# 실행
# ---------------------------
def main():

    result = {}

    for category_ko, category_en in CATEGORIES.items():

     print(f"\n뉴스 수집중: {category_ko}")

     articles = fetch_news(category_en)

     print("토론 주제 생성중...")

     topics = generate_topics(category_ko, articles)

     result[category_ko] = topics

    os.makedirs("data", exist_ok=True)

    with open("data/topics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n완료! data/topics.json 생성")


if __name__ == "__main__":
    main()