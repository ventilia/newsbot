import asyncio
from datetime import timedelta, datetime

import aiohttp
from bs4 import BeautifulSoup
import feedparser
import logging
from groq import Groq
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse, quote_plus
from config.settings import GROQ_API_KEY, DEFAULT_AI_MODEL

logger = logging.getLogger(__name__)


class RSSFinder:
    """Поиск RSS-лент по теме с использованием Groq API."""

    def __init__(self):

        if not GROQ_API_KEY:
            logger.critical("GROQ_API_KEY не найден в настройках!")
            raise ValueError("GROQ_API_KEY is required")

        self.client = Groq(api_key=GROQ_API_KEY)
        self.search_url = "https://duckduckgo.com/html/?q={}"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        logger.info("RSSFinder инициализирован")

    async def find_rss_by_topic(self, topic: str) -> List[Dict]:
        """Находит RSS-ленты по заданной теме."""
        logger.info(f"=== НАЧАЛО ПОИСКА RSS-ЛЕНТ ДЛЯ ТЕМЫ: {topic} ===")
        start_time = asyncio.get_event_loop().time()

        try:
            search_queries = await self.generate_search_keywords(topic)
            logger.info(f"Сгенерировано поисковых запросов: {len(search_queries)}")
            logger.debug(f"Поисковые запросы: {search_queries}")

            sites = set()
            async with aiohttp.ClientSession(headers=self.headers) as session:
                for query in search_queries:
                    try:
                        url = self.search_url.format(quote_plus(query))
                        logger.debug(f"Запрос к DuckDuckGo: {url}")

                        async with session.get(url, timeout=15) as response:
                            if response.status != 200:
                                logger.warning(f"DDG вернул статус {response.status} для запроса '{query}'")
                                continue

                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')

                            found_links = 0
                            for link in soup.select('a.result__a'):
                                href = link.get('href')
                                if href and href.startswith('http'):
                                    domain = urlparse(href).netloc
                                    if domain:
                                        sites.add(f"https://{domain}")
                                        found_links += 1

                            logger.info(f"Найдено {found_links} сайтов для запроса '{query}'")

                    except asyncio.TimeoutError:
                        logger.warning(f"Таймаут при поиске по запросу: {query}")
                    except Exception as e:
                        logger.error(f"Ошибка при поиске по запросу '{query}': {str(e)}", exc_info=True)
                        continue

            logger.info(f"Всего найдено уникальных сайтов: {len(sites)}")

            if not sites:
                logger.warning("Не найдено сайтов для поиска RSS")
                return []

            # Ограничиваем количество сайтов для обработки
            sites_list = list(sites)[:20]
            logger.info(f"Обработка RSS для {len(sites_list)} сайтов")

            feeds = await self.discover_rss_feeds(sites_list)
            logger.info(f"Найдено RSS-лент: {len(feeds)}")

            validated = await self.validate_feeds(feeds)
            logger.info(f"Валидных RSS-лент: {len(validated)}")


            sorted_feeds = sorted(validated, key=lambda x: x.get('entry_count', 0), reverse=True)[:5]
            logger.info(f"Возвращено топ-{len(sorted_feeds)} RSS-лент")

            execution_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"=== ПОИСК RSS-ЛЕНТ ЗАВЕРШЕН (время выполнения: {execution_time:.2f} сек) ===")

            return sorted_feeds

        except Exception as e:
            logger.error(f"Критическая ошибка в find_rss_by_topic: {str(e)}", exc_info=True)
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"=== ПОИСК RSS-ЛЕНТ ПРЕРВАН ИЗ-ЗА ОШИБКИ (время выполнения: {execution_time:.2f} сек) ===")
            return []

    async def generate_search_keywords(self, topic: str) -> List[str]:

        logger.info(f"Генерация поисковых запросов для темы: {topic}")

        prompt = (
            f"Сгенерируй 3 различных поисковых запроса на РУССКОМ языке для поиска RSS-лент и блогов на тему '{topic}'. "
            "Запросы должны быть конкретными, содержать слова 'RSS', 'лента', 'подписка'. "
            "Верни только запросы, каждый с новой строки, без нумерации и дополнительного текста."
        )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Используем актуальную модель
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150,
                timeout=20
            ))

            if not response or not response.choices:
                logger.error("Пустой ответ от Groq при генерации запросов")
                raise ValueError("Empty response from Groq")

            content = response.choices[0].message.content.strip()
            logger.debug(f"Ответ Groq для генерации запросов: {content}")


            keywords = [k.strip() for k in content.split('\n') if k.strip() and len(k.strip()) > 5]
            logger.info(f"Сгенерировано ключевых слов: {len(keywords)}")


            if len(keywords) < 2:
                logger.warning("Не удалось сгенерировать достаточно ключевых слов, используем резервные")
                return [
                    f'"{topic}" RSS лента новости',
                    f'"{topic}" официальный блог подписка',
                    f'"{topic}" обновления RSS'
                ]

            return keywords[:3]

        except Exception as e:
            logger.error(f"Ошибка при генерации ключевых слов: {str(e)}", exc_info=True)
            logger.warning("Используем резервные поисковые запросы")
            return [
                f'"{topic}" RSS лента новости',
                f'"{topic}" официальный блог подписка',
                f'"{topic}" обновления RSS'
            ]

    async def discover_rss_feeds(self, sites: List[str]) -> List[Dict]:

        logger.info(f"Поиск RSS-лент на {len(sites)} сайтах")
        feeds = []

        async with aiohttp.ClientSession(headers=self.headers, timeout=aiohttp.ClientTimeout(total=10)) as session:
            tasks = [self.fetch_site_feeds(site, session) for site in sites]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for site, result in zip(sites, results):
                if isinstance(result, Exception):
                    logger.error(f"Ошибка при обработке сайта {site}: {str(result)}")
                    continue
                feeds.extend(result)


        unique_feeds = list({feed['url']: feed for feed in feeds}.values())
        logger.info(f"Всего найдено RSS-лент (после удаления дубликатов): {len(unique_feeds)}")
        return unique_feeds

    async def fetch_site_feeds(self, site_url: str, session: aiohttp.ClientSession) -> List[Dict]:

        found_feeds = []
        logger.debug(f"Проверка сайта на наличие RSS: {site_url}")

        try:
            async with session.get(site_url, timeout=10) as response:
                if response.status != 200:
                    logger.warning(f"Сайт {site_url} вернул статус {response.status}")
                    return found_feeds

                html = await response.text()
                soup = BeautifulSoup(html, 'lxml')


                for link in soup.find_all('link', {
                    'type': ['application/rss+xml', 'application/atom+xml', 'application/feed+json']}):
                    href = link.get('href')
                    if href:
                        feed_url = urljoin(site_url, href)
                        title = link.get('title', soup.title.string if soup.title else site_url)
                        found_feeds.append({
                            'url': feed_url,
                            'title': title.strip(),
                            'site': site_url
                        })
                        logger.debug(f"Найдена RSS-лента: {feed_url}")


                if not found_feeds:
                    common_paths = ['/rss', '/feed', '/rss.xml', '/feed.xml', '/atom.xml', '/blog/rss', '/news/rss']
                    for path in common_paths:
                        feed_url = urljoin(site_url, path)
                        try:
                            async with session.head(feed_url, timeout=5) as feed_response:
                                content_type = feed_response.headers.get('content-type', '').lower()
                                if feed_response.status == 200 and (
                                        'xml' in content_type or 'json' in content_type or 'feed' in content_type):
                                    found_feeds.append({
                                        'url': feed_url,
                                        'title': f"{site_url} RSS",
                                        'site': site_url
                                    })
                                    logger.debug(f"Найдена RSS-лента по общему пути: {feed_url}")
                                    break
                        except Exception as e:
                            logger.debug(f"Ошибка при проверке пути {feed_url}: {str(e)}")
                            continue

                logger.info(f"На сайте {site_url} найдено RSS-лент: {len(found_feeds)}")
                return found_feeds

        except asyncio.TimeoutError:
            logger.warning(f"Таймаут при загрузке сайта: {site_url}")
        except Exception as e:
            logger.error(f"Ошибка при обработке сайта {site_url}: {str(e)}", exc_info=True)

        return found_feeds

    async def validate_feeds(self, feeds: List[Dict]) -> List[Dict]:

        logger.info(f"Валидация {len(feeds)} RSS-лент")
        validated = []

        tasks = [self.parse_and_validate(feed) for feed in feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for feed, result in zip(feeds, results):
            if isinstance(result, Exception):
                logger.error(f"Ошибка валидации ленты {feed['url']}: {str(result)}")
                continue
            if result:
                validated.append(result)

        logger.info(f"Успешно валидировано RSS-лент: {len(validated)}")
        return validated

    async def parse_and_validate(self, feed: Dict) -> Dict | None:

        try:
            logger.debug(f"Парсинг RSS-ленты: {feed['url']}")
            parsed = feedparser.parse(feed['url'])

            if not parsed or not parsed.entries:
                logger.warning(f"RSS-лента {feed['url']} не содержит записей")
                return None


            now = datetime.now()
            month_ago = now - timedelta(days=30)
            recent_entries = 0

            for entry in parsed.entries[:10]:
                try:
                    pub_date = entry.get('published_parsed') or entry.get('updated_parsed')
                    if pub_date:
                        entry_date = datetime.fromtimestamp(pub_date.timestamp())
                        if entry_date > month_ago:
                            recent_entries += 1
                except:
                    continue


            if recent_entries < 2:
                logger.warning(f"RSS-лента {feed['url']} имеет недостаточно свежих записей: {recent_entries}")
                return None


            feed['title'] = parsed.feed.get('title', feed['title'])
            feed['entry_count'] = len(parsed.entries)
            feed['last_updated'] = parsed.feed.get('updated', parsed.entries[0].get('published', ''))

            logger.info(
                f"RSS-лента {feed['url']} успешно валидирована. Записей: {feed['entry_count']}, Свежих: {recent_entries}")
            return feed

        except Exception as e:
            logger.error(f"Ошибка парсинга RSS {feed['url']}: {str(e)}", exc_info=True)
            return None