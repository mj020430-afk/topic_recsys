import json
import os
from typing import Dict, List, Any

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

if not NEWS_API_KEY:
    raise ValueError("NEWS_API_KEY가 설정되지 않았습니다.")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

client = OpenAI(api_key=OPENAI_API_KEY)

CATEGORIES = {
    "기술/AI": "technology",
    "경제/산업": "business",
    "정치/사회": "general",
    "과학/환경": "science",
}


def fetch_news(category_en: str, page_size: int = 10) -> List[Dict[str, Any]]:
    query_map = {
        "technology": "AI OR artificial intelligence OR technology OR software",
        "business": "economy OR business OR industry OR market OR investment",
        "general": "politics OR government OR society OR policy OR public",
        "science": "science OR climate OR environment OR energy OR sustainability",
    }

    query = query_map[category_en]

    url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": NEWS_API_KEY,
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "ok":
        raise ValueError(f"NewsAPI 요청 실패: {data}")

    articles = []
    for article in data.get("articles", []):
        title = (article.get("title") or "").strip()
        description = (article.get("description") or "").strip()
        url = article.get("url", "")
        source = (article.get("source") or {}).get("name", "")
        published_at = article.get("publishedAt", "")

        if not title:
            continue

        articles.append(
            {
                "title": title,
                "description": description,
                "url": url,
                "source": source,
                "publishedAt": published_at,
            }
        )

    return articles


def build_generation_prompt(category_ko: str, articles: List[Dict[str, Any]], excluded_topics: List[str]) -> str:
    article_lines = []
    for idx, article in enumerate(articles[:8], start=1):
        article_lines.append(
            f"{idx}. 제목: {article['title']}\n"
            f"   설명: {article.get('description', '')}\n"
            f"   출처: {article.get('source', '')}"
        )

    excluded_text = "\n".join([f"- {t}" for t in excluded_topics]) if excluded_topics else "없음"
    news_text = "\n".join(article_lines)

    return f"""
다음은 '{category_ko}' 분야의 최신 뉴스이다.

{news_text}

이 뉴스를 바탕으로 실제 토론 가능한 논제 후보를 정확히 6개 생성하라.

중요 조건:
1. 반드시 한국어
2. 반드시 '{category_ko}' 카테고리와 직접 관련된 주제만 생성
3. 지나치게 폭력적, 혐오적, 차별적, 선정적, 범죄 조장적인 주제는 금지
4. 학술적, 정책적, 사회적 토론에 적합한 수준으로 작성
5. 각 논제는 아래 두 유형 중 하나
   - "pro_con": ~해야 하는가?
   - "choice_ab": A와 B 중 무엇을 우선해야 하는가?
6. 반드시 아래 필드 포함
   - topic
   - debate_type
   - side_1
   - side_2
7. 아래 이미 생성되었거나 제외된 논제와 중복되지 않게 작성하라
8. 같은 핵심 이슈를 여러 카테고리에 중복 생성하지 마라.
9. 각 카테고리마다 가장 대표적인 서로 다른 주제를 우선 생성하라.

이미 제외/선정된 논제:
{excluded_text}

출력 형식:
{{
  "topics": [
    {{
      "topic": "...",
      "debate_type": "pro_con",
      "side_1": "...",
      "side_2": "..."
    }}
  ]
}}

반드시 JSON만 출력하라.
""".strip()


def build_validation_prompt(category_ko: str, topics: List[Dict[str, Any]]) -> str:
    topics_json = json.dumps(topics, ensure_ascii=False, indent=2)

    return f"""
너는 멀티에이전트 토론 시스템의 주제 검증기다.

아래는 '{category_ko}' 카테고리의 토론 논제 후보들이다.

{topics_json}

각 후보에 대해 다음 기준으로 검증하라.

검증 기준:
1. 해당 카테고리와 직접 관련 있는가
2. 찬반 또는 A/B 토론 구조가 명확한가
3. 지나치게 폭력적, 혐오적, 차별적, 선정적, 범죄 조장적이지 않은가
4. 자극적 선동보다 공적 토론 주제로 적절한가
5. 너무 민감하거나 비윤리적이어서 추천에서 제외해야 하는가
6. 문장이 모호하거나 설명형 질문은 아닌가

출력 규칙:
- 각 topic마다 keep 값은 true 또는 false
- reason에는 판정 이유를 짧게 작성
- 통과 가능한 주제만 true 처리

출력 형식:
{{
  "results": [
    {{
      "topic": "...",
      "keep": true,
      "reason": "..."
    }}
  ]
}}

반드시 JSON만 출력하라.
""".strip()


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError(f"JSON 파싱 실패: {text}")


def normalize_topics(raw_topics: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized = []

    for item in raw_topics:
        topic = str(item.get("topic", "")).strip()
        debate_type = str(item.get("debate_type", "")).strip()
        side_1 = str(item.get("side_1", "")).strip()
        side_2 = str(item.get("side_2", "")).strip()

        if not topic:
            continue
        if debate_type not in {"pro_con", "choice_ab"}:
            continue
        if not side_1 or not side_2:
            continue

        normalized.append(
            {
                "topic": topic,
                "debate_type": debate_type,
                "side_1": side_1,
                "side_2": side_2,
            }
        )

    return normalized


def rule_based_filter(topics: List[Dict[str, str]]) -> List[Dict[str, str]]:
    banned_keywords = [
        "자살", "자해", "학살", "테러", "성착취", "아동 포르노",
        "마약 제조", "살인", "증오", "인종청소", "보복 폭력"
    ]

    filtered = []

    for item in topics:
        topic_text = item["topic"]

        if any(keyword in topic_text for keyword in banned_keywords):
            continue

        if item["debate_type"] == "pro_con" and "해야 하는가" not in topic_text:
            continue

        if item["debate_type"] == "choice_ab":
            if "무엇을 우선" not in topic_text and "중" not in topic_text:
                continue

        filtered.append(item)

    return filtered


def generate_topic_candidates(category_ko: str, articles: List[Dict[str, Any]], excluded_topics: List[str]) -> List[Dict[str, str]]:
    if not articles:
        return []

    prompt = build_generation_prompt(category_ko, articles, excluded_topics)

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt
    )

    parsed = extract_json(response.output_text)
    topics = parsed.get("topics", [])

    return normalize_topics(topics)


def validate_topics_with_llm(category_ko: str, topics: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    if not topics:
        return []

    prompt = build_validation_prompt(category_ko, topics)

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt
    )

    parsed = extract_json(response.output_text)
    return parsed.get("results", [])


def select_passed_topics(candidates: List[Dict[str, str]], validations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    validation_map = {
        item.get("topic", "").strip(): item
        for item in validations
    }

    passed = []
    for candidate in candidates:
        result = validation_map.get(candidate["topic"])
        if result and result.get("keep") is True:
            passed.append(candidate)

    return passed


def deduplicate_topics(topics: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    unique = []

    for item in topics:
        key = item["topic"].strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return unique


def generate_and_validate_topics(category_ko: str, articles: List[Dict[str, Any]], target_count: int = 2) -> Dict[str, Any]:
    final_topics: List[Dict[str, str]] = []
    excluded_topics: List[str] = []

    max_rounds = 3

    for round_idx in range(max_rounds):
        print(f"[DEBUG] {category_ko} 생성/검증 라운드: {round_idx + 1}")

        candidates = generate_topic_candidates(category_ko, articles, excluded_topics)
        candidates = rule_based_filter(candidates)
        candidates = deduplicate_topics(candidates)

        print(f"[DEBUG] {category_ko} 후보 수: {len(candidates)}")

        validations = validate_topics_with_llm(category_ko, candidates)
        passed = select_passed_topics(candidates, validations)

        print(f"[DEBUG] {category_ko} 검증 통과 수: {len(passed)}")

        final_topics.extend(passed)
        final_topics = deduplicate_topics(final_topics)

        if len(final_topics) >= target_count:
            break

        excluded_topics.extend([item["topic"] for item in candidates])

    if len(final_topics) < target_count:
        print(f"[DEBUG] {category_ko} 검증 통과 부족, 후보 보완")
        backup_candidates = generate_topic_candidates(category_ko, articles, excluded_topics)
        backup_candidates = rule_based_filter(backup_candidates)
        backup_candidates = deduplicate_topics(backup_candidates)

        existing = {item["topic"] for item in final_topics}
        for item in backup_candidates:
            if item["topic"] not in existing:
                final_topics.append(item)
                existing.add(item["topic"])
            if len(final_topics) >= target_count:
                break

    return {"topics": final_topics[:target_count]}


def main() -> None:
    result = {}

    for category_ko, category_en in CATEGORIES.items():
        print(f"\n[1] 뉴스 수집중: {category_ko}")
        articles = fetch_news(category_en)
        print(f"[2] 수집 기사 수: {len(articles)}")

        if articles:
            print(f"[3] 첫 기사 제목: {articles[0]['title']}")
        else:
            print("[3] 기사 없음")

        print("[4] 주제 생성 + 검증중...")
        topic_result = generate_and_validate_topics(category_ko, articles, target_count=2)
        print(f"[5] 최종 결과 개수: {len(topic_result['topics'])}")

        result[category_ko] = topic_result

    os.makedirs("data", exist_ok=True)

    output_path = "data/topics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n완료! {output_path} 생성")


if __name__ == "__main__":
    main()