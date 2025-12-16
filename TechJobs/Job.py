import scrapy
from scrapy.crawler import CrawlerProcess
import json
import os


class HhUzAPISpider(scrapy.Spider):
    name = "hh_uz_api"

    def __init__(self, area=97, role=156, max_pages=30, **kwargs):
        super().__init__(**kwargs)
        self.area = area
        self.role = role
        self.max_pages = int(max_pages)
        self.results = []

    def start_requests(self):
        base_url = "https://api.hh.ru/vacancies"

        for page in range(self.max_pages):
            params = {
                "area": self.area,
                "professional_role": self.role,
                "page": page,
                "per_page": 100
            }

            yield scrapy.Request(
                url=f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}",
                callback=self.parse_api
            )

    def parse_api(self, response):
        data = json.loads(response.text)
        items = data.get("items", [])

        # 🛑 Stop if API returns empty page
        if not items:
            self.logger.info("No more vacancies found, stopping pagination.")
            return

        for item in items:
            salary = item.get("salary")

            self.results.append({
                "title": item.get("name"),
                "company": item.get("employer", {}).get("name"),
                "location": item.get("area", {}).get("name"),
                "link": item.get("alternate_url"),
                "salary_from": salary.get("from") if salary else None,
                "salary_to": salary.get("to") if salary else None,
                "currency": salary.get("currency") if salary else None
            })

    def close(self, reason):
        os.makedirs("data", exist_ok=True)
        output_path = os.path.join("vacancies.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Saved {len(self.results)} vacancies to {output_path}")


if __name__ == "__main__":
    process = CrawlerProcess(settings={
        "LOG_LEVEL": "INFO",
        "USER_AGENT": "HH-API-Scraper/1.0",
        "DOWNLOAD_DELAY": 0.2
    })

    process.crawl(HhUzAPISpider, max_pages=30)
    process.start()
