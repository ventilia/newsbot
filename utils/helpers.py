import re
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def sanitize_html(text: str) -> str:

    try:
        allowed_tags = ['b', 'i', 'u', 's', 'code', 'pre', 'a']

        pattern = r'<(?!/?({}))([^>]*)>'.format('|'.join(allowed_tags))
        cleaned = re.sub(pattern, '', text)


        cleaned = re.sub(r'<a[^>]*>', '<a>', cleaned)

        cleaned = re.sub(r'\s+', ' ', cleaned).strip()


        cleaned = cleaned.replace('\\n', '\n')
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        logger.debug(f"HTML очищен. Длина до: {len(text)}, после: {len(cleaned)}")
        return cleaned
    except Exception as e:
        logger.error(f"Ошибка при очистке HTML: {str(e)}", exc_info=True)
        return text


def generate_post_hash(content: str) -> str:

    try:
        if not content:
            return ""

        sample = content[:500].lower().strip()
        hash_value = hashlib.md5(sample.encode('utf-8')).hexdigest()
        logger.debug(f"Сгенерирован хеш для контента: {hash_value[:8]}...")
        return hash_value
    except Exception as e:
        logger.error(f"Ошибка при генерации хеша: {str(e)}", exc_info=True)
        return ""


def clean_rss_content(html_content: str) -> str:

    try:
        if not html_content:
            return ""

        logger.debug(f"Очистка RSS контента. Длина: {len(html_content)}")


        soup = BeautifulSoup(html_content, 'html.parser')


        for element in soup(['script', 'style', 'noscript', 'iframe', 'object', 'embed']):
            element.decompose()


        for comment in soup.find_all(text=lambda text: isinstance(text, str) and text.startswith('<!--')):
            comment.extract()


        text = soup.get_text(separator=' ', strip=True)


        text = re.sub(r'\s+', ' ', text).strip()


        max_length = 2000
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.debug(f"Контент обрезан до {max_length} символов")

        logger.debug(f"Очистка RSS контента завершена. Длина: {len(text)}")
        return text

    except Exception as e:
        logger.error(f"Ошибка при очистке RSS контента: {str(e)}", exc_info=True)
        return html_content[:1000] if html_content else ""


def split_long_message(text: str, max_length: int = 4000) -> List[str]:

    if not text or len(text) <= max_length:
        return [text]

    logger.info(f"Разбиение сообщения длиной {len(text)} на части по {max_length} символов")

    parts = []
    current_part = ""


    paragraphs = text.split('\n\n')

    for paragraph in paragraphs:
        if len(current_part) + len(paragraph) + 2 <= max_length:
            if current_part:
                current_part += "\n\n"
            current_part += paragraph
        else:
            if current_part:
                parts.append(current_part)
                current_part = paragraph
            else:

                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_sentence_part = ""
                for sentence in sentences:
                    if len(current_sentence_part) + len(sentence) + 1 <= max_length:
                        if current_sentence_part:
                            current_sentence_part += " "
                        current_sentence_part += sentence
                    else:
                        if current_sentence_part:
                            parts.append(current_sentence_part)
                        current_sentence_part = sentence
                if current_sentence_part:
                    current_part = current_sentence_part

    if current_part:
        parts.append(current_part)

    logger.info(f"Сообщение разбито на {len(parts)} частей")
    return parts


async def retry_async(func, max_attempts: int = 3, delay: int = 1):

    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            logger.warning(f"Попытка {attempt + 1}/{max_attempts} не удалась: {str(e)}")
            if attempt == max_attempts - 1:
                logger.error("Все попытки исчерпаны", exc_info=True)
                raise
            await asyncio.sleep(delay * (2 ** attempt))


def extract_domain(url: str) -> str:

    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        logger.debug(f"Извлечен домен из {url}: {domain}")
        return domain
    except Exception as e:
        logger.error(f"Ошибка при извлечении домена из {url}: {str(e)}", exc_info=True)
        return url


def format_time_delta(delta: timedelta) -> str:

    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, _ = divmod(remainder, 60)

    parts = []
    if days:
        parts.append(f"{days}д")
    if hours:
        parts.append(f"{hours}ч")
    if minutes:
        parts.append(f"{minutes}м")

    return ' '.join(parts) or '0м'