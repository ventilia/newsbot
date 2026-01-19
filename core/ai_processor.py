import asyncio
import random
import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse
from groq import Groq, GroqError
from config.settings import GROQ_API_KEY, DEFAULT_AI_MODEL, GROQ_MODELS
from utils.helpers import sanitize_html, clean_rss_content

logger = logging.getLogger(__name__)


def is_http_url(s: str) -> bool:

    try:
        return urlparse(s).scheme in {"http", "https"}
    except Exception:
        return False


def md_to_html(text: str) -> str:

    try:
        import markdown2
    except ImportError:
        return text

    html = markdown2.markdown(text)
    html = html.replace("<strong>", "<b>").replace("</strong>", "</b>").replace("<em>", "<i>").replace("</em>", "</i>")
    html = re.sub(r"</?p>", "", html)
    return html


class AIProcessor:


    SAFE_MODEL = "llama-3.1-8b-instant"
    SUPPORTED_MODELS = set(GROQ_MODELS + ["gpt-4o-mini", "gpt-4"])

    def __init__(self) -> None:

        if not GROQ_API_KEY:
            logger.critical("GROQ_API_KEY не найден в настройках!")
            raise ValueError("GROQ_API_KEY is required")

        self.client = Groq(api_key=GROQ_API_KEY)
        self.emojis = {
            "tech": ["💻", "🚀", "🔧", "⚡", "🌐", "📱", "🤖"],
            "news": ["📰", "🗞️", "🔥", "⚠️", "💡", "✨", "🎯"],
            "business": ["💼", "📈", "💰", "🏢", "📊", "🤝", "💵"],
            "entertainment": ["🎬", "🎭", "🎪", "🎨", "🎤", "🎧", "🌟"],
            "sports": ["⚽", "🏀", "🎾", "🏃", "🏊", "🏋️", "🏆"],
            "politics": ["🏛️", "⚖️", "🌍", "🤝", "📜", "🗳️", "🎖️"],
            "science": ["🔬", "🔭", "🧪", "🧠", "🧬", "🌱", "⚙️"]
        }
        logger.info(f"AIProcessor инициализирован. Модель по умолчанию: {self.SAFE_MODEL}")

    async def process_content(self, entry: Dict, ch_settings: Dict) -> str:

        try:
            model = ch_settings.get("ai_model") or self.SAFE_MODEL
            if model not in self.SUPPORTED_MODELS:
                logger.warning(
                    f"Модель {model} не поддерживается. Доступные модели: {', '.join(self.SUPPORTED_MODELS)}")
                logger.info(f"Используем модель по умолчанию: {self.SAFE_MODEL}")
                model = self.SAFE_MODEL

            topic = ch_settings.get("topic", "новости")
            sys_prompt = (ch_settings.get("ai_prompt") or self._default_prompt().format(topic=topic))


            clean_content = clean_rss_content(entry['content'])
            user_prompt = f"Переработай эту новость в пост для Telegram (700-900 символов): Title: {entry['title']}. Content: {clean_content[:700]}"

            logger.info(f"Запрос к Groq API для обработки контента. Модель: {model}, Тема: {topic}")
            logger.debug(f"System prompt (первые 100 символов): {sys_prompt[:100]}...")
            logger.debug(f"User prompt (первые 100 символов): {user_prompt[:100]}...")

            raw_response = await self._call_groq(model, sys_prompt, user_prompt)


            if not raw_response or len(raw_response.strip()) < 100:
                logger.warning(
                    f"Получен короткий ответ от Groq ({len(raw_response.strip())} символов), используем улучшенный fallback")
                return await self._enhanced_fallback_format(entry, topic)


            if any(prompt_word in raw_response.lower() for prompt_word in
                   ["system:", "user:", "assistant:", "instruct", "you are", "твоя задача", "правила:", "пример:",
                    "формат:"]):
                logger.warning("В ответе обнаружены признаки промпта, используем улучшенный fallback")
                return await self._enhanced_fallback_format(entry, topic)


            final_post = self._guaranteed_formatting(raw_response, topic)
            logger.info(f"Успешно обработан контент для поста. Длина: {len(final_post)} символов")
            logger.debug(f"Финальный пост: {final_post}")
            return final_post[:1500]  # Увеличенное ограничение длины

        except Exception as e:
            logger.error(f"Ошибка при обработке контента: {str(e)}", exc_info=True)
            logger.info("Используем улучшенное fallback форматирование")
            return await self._enhanced_fallback_format(entry, topic)

    async def simple_translate(self, text: str) -> str:

        try:
            if not text or len(text.strip()) < 3:
                return text

            logger.info(f"Перевод текста длиной {len(text)} символов")
            response = await self._call_groq(
                self.SAFE_MODEL,
                "You are a professional translator. Translate the text to Russian accurately and naturally. Keep company names and product names untranslated. Return ONLY the translated text without any additional comments.",
                f"Translate to Russian: {text[:500]}"
            )
            return response.strip() if response else text
        except Exception as e:
            logger.error(f"Ошибка при переводе: {str(e)}", exc_info=True)
            return text

    async def _call_groq(self, model: str, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:

        retry_delay = 1  # секунд

        for attempt in range(max_retries):
            try:
                logger.debug(f"Попытка {attempt + 1}/{max_retries} вызова Groq API с моделью {model}")

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, lambda: self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.4,
                    max_tokens=1200,
                    top_p=0.9,
                    timeout=45
                ))

                if not response or not response.choices:
                    raise ValueError("Пустой ответ от Groq API")

                result = response.choices[0].message.content.strip()
                logger.debug(f"Получен ответ от Groq (первые 200 символов): {result[:200]}...")
                return result

            except GroqError as e:
                error_msg = str(e)
                logger.error(f"Groq API ошибка на попытке {attempt + 1}: {error_msg}")
                if "rate_limit" in error_msg.lower() and attempt < max_retries - 1:
                    logger.warning(f"Достигнут лимит запросов, ждем {retry_delay} секунд")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                # Попытка сменить модель при ошибке
                if "model_decommissioned" in error_msg.lower() and attempt == 0:
                    logger.warning(f"Модель {model} устарела, пробуем использовать {self.SAFE_MODEL}")
                    model = self.SAFE_MODEL
                    continue
                raise
            except Exception as e:
                logger.error(f"Неожиданная ошибка на попытке {attempt + 1}: {str(e)}", exc_info=True)
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                raise

        logger.error("Все попытки вызова Groq API исчерпаны")
        raise Exception("Max retries exceeded for Groq API call")

    async def _enhanced_fallback_format(self, entry: Dict, topic: str) -> str:

        logger.info("Используется УЛУЧШЕННОЕ резервное форматирование контента")
        try:

            title_ru = await self.simple_translate(entry['title'])


            clean_content = clean_rss_content(entry['content'])
            cont_ru = await self.simple_translate(clean_content)
            cont_ru = cont_ru.replace("\\n", "\n").strip()


            paragraphs = [p.strip() for p in cont_ru.split('\n') if p.strip()]


            formatted_paragraphs = []
            current_para = ""

            for para in paragraphs:
                if len(para) < 30:
                    current_para += para + " "
                else:
                    if current_para:
                        formatted_paragraphs.append(current_para.strip())
                        current_para = ""
                    formatted_paragraphs.append(para)

            if current_para:
                formatted_paragraphs.append(current_para.strip())


            description_paragraphs = formatted_paragraphs[:3]
            description = "\n\n".join(description_paragraphs)


            if len(description) < 200:

                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cont_ru) if s.strip()]
                if len(sentences) > 3:
                    description = sentences[0] + " " + sentences[1] + "\n\n" + sentences[2]


            if len(description) > 800:
                description = description[:800] + "..."

            emoji = random.choice(self._emojis_for(topic))
            hashtags = " ".join(self._hashtags_for(topic))
            clean_title = title_ru.strip(' "\'')


            result = (
                f"<b>{emoji} {clean_title}</b>\n\n"
                f"{description}\n\n"
                f"{hashtags}"
            )

            logger.info(f"Улучшенное резервное форматирование завершено. Длина: {len(result)}")
            logger.debug(f"Резервный пост: {result}")
            return result[:1500]
        except Exception as e:
            logger.error(f"Ошибка в улучшенном резервном форматировании: {str(e)}", exc_info=True)

            return (
                f"<b>⚠️ {entry.get('title', 'Новость')}</b>\n\n"
                f"{entry.get('content', '')[:500]}...\n\n"
                f"#новости #аварийныйрежим"
            )

    def _guaranteed_formatting(self, raw_text: str, topic: str) -> str:

        try:
            text = raw_text.strip()
            logger.debug(
                f"Начало гарантированного форматирования. Исходный текст (первые 200 символов): {text[:200]}...")


            lines = [line.strip() for line in text.split('\n') if line.strip()]

            title = ""
            content_lines = []

            if lines:

                first_line = lines[0]


                first_line = re.sub(r'^[#\*]+\s*', '', first_line)
                first_line = re.sub(r'[\*#_]+$', '', first_line)


                if '<b>' in first_line and '</b>' in first_line:
                    title_match = re.search(r'<b>(.*?)</b>', first_line)
                    if title_match:
                        title = title_match.group(1).strip()
                else:
                    title = first_line


                title = re.sub(r'^[^\wа-яА-ЯёЁ]+', '', title)
                title = re.sub(r'[^\wа-яА-ЯёЁ\s.,!?;:()\-\"\'—–-]+$', '', title)


                content_lines = lines[1:]


            if not title and content_lines:
                for i, line in enumerate(content_lines):
                    if len(line) < 100 and (
                            line.endswith('.') or line.endswith('!') or line.endswith('?') or len(line) < 50):
                        title = line
                        content_lines = content_lines[i + 1:]
                        break


            if not title:
                first_sentence = re.split(r'[.!?]', text)[0].strip()
                if len(first_sentence) > 20 and len(first_sentence) < 100:
                    title = first_sentence + ("" if first_sentence.endswith(("?", "!", ".")) else ".")
                else:
                    title = "Новости по теме: " + topic


            content = "\n".join(content_lines).strip()

            paragraphs = []


            raw_paragraphs = re.split(r'\n{2,}', content)

            if len(raw_paragraphs) < 2:

                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if s.strip()]
                if sentences:

                    para_sentences = []
                    for sentence in sentences:
                        para_sentences.append(sentence)
                        if len(para_sentences) >= 2 or (len(para_sentences) == 1 and len(sentence) > 150):
                            paragraphs.append(" ".join(para_sentences))
                            para_sentences = []
                    if para_sentences:
                        paragraphs.append(" ".join(para_sentences))
            else:
                paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]


            paragraphs = paragraphs[:3]


            clean_paragraphs = []
            for para in paragraphs:

                para = re.sub(r'^[\-\•\*]\s*', '', para)

                para = re.sub(r'^\d+[\.\)]\s*', '', para)

                para = re.sub(r'\s+', ' ', para).strip()
                if para and len(para) > 30:
                    clean_paragraphs.append(para)


            if not clean_paragraphs:
                first_part = re.split(r'[.!?]', content)[0].strip()
                if len(first_part) > 50:
                    clean_paragraphs.append(first_part + ".")
                else:
                    clean_paragraphs.append(content[:300] + ("..." if len(content) > 300 else ""))


            emoji = random.choice(self._emojis_for(topic))


            if not re.match(
                    r'^[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0]',
                    title):
                title = f"{emoji} {title}"


            hashtags = " ".join(self._hashtags_for(topic))


            final_post = f"<b>{title}</b>\n\n"
            final_post += "\n\n".join(clean_paragraphs) + "\n\n"
            final_post += hashtags


            final_post = re.sub(r'\n{3,}', '\n\n', final_post)
            final_post = re.sub(r' +', ' ', final_post)
            final_post = final_post.strip()

            logger.info(f"Гарантированное форматирование завершено. Длина: {len(final_post)} символов")
            logger.debug(f"Отформатированный пост (первые 300 символов): {final_post[:300]}...")
            return final_post

        except Exception as e:
            logger.error(f"Ошибка в гарантированном форматировании: {str(e)}", exc_info=True)
            logger.warning("Используем простое форматирование")

            emoji = random.choice(self._emojis_for(topic))
            hashtags = " ".join(self._hashtags_for(topic))
            clean_text = re.sub(r'\s+', ' ', raw_text.strip())
            return f"<b>{emoji} Новость</b>\n\n{clean_text[:600]}...\n\n{hashtags}"

    def _emojis_for(self, topic: str) -> List[str]:

        t = topic.lower()
        if any(w in t for w in
               ("политик", "государств", "власть", "президент", "минист", "дипломат", "альянс", "союз")):
            return self.emojis["politics"]
        if any(w in t for w in ("развлечени", "кино", "музык", "звезд", "шоу", "юмор", "сатир")):
            return self.emojis["entertainment"]
        if any(w in t for w in ("спорт", "матч", "чемпион", "турнир", "игрок", "команда")):
            return self.emojis["sports"]
        if any(w in t for w in
               ("it", "tech", "технолог", "программ", "код", "разработка", "робот", "наука", "исследован", "открыт")):
            return self.emojis["science"]
        if any(w in t for w in ("бизнес", "финанс", "экономик", "рынок", "маркет", "стартап", "компания", "корпораци")):
            return self.emojis["business"]
        return self.emojis["news"]

    def _hashtags_for(self, topic: str) -> List[str]:

        base = {
            "политика": ["#политика", "#геополитика", "#международныеотношения"],
            "россия": ["#россия", "#российскаяполитика", "#новостироссии"],
            "сша": ["#сша", "#америка", "#внешняяполитика"],
            "европа": ["#европа", "#евросоюз", "#европейскаяполитика"],
            "украина": ["#украина", "#киев", "#киевскийрежим"],
            "германия": ["#германия", "#берлин", "#немецкаяполитика"],
            "польша": ["#польша", "#варшава", "#польскаяполитика"],
            "военные": ["#армия", "#вооруженныесилы", "#оборона"],
            "дипломатия": ["#дипломатия", "#переговоры", "#мирныепроцессы"],
            "наука": ["#наука", "#технологии", "#инновации"],
            "экономика": ["#экономика", "#финансы", "#бизнес"],
            "культура": ["#культура", "#искусство", "#история"],
            "спорт": ["#спорт", "#чемпионат", "#олимпиада"],
            "здоровье": ["#здоровье", "#медицина", "#образжизни"]
        }

        t = topic.lower()
        for k, tags in base.items():
            if k in t:
                return random.sample(tags, min(3, len(tags)))


        clean_topic = re.sub(r'[^\w\s]', '', topic.lower()).strip()
        clean_words = clean_topic.split()

        hashtags = ["#новости"]

        if clean_words:
            main_word = clean_words[0]
            if len(main_word) > 3 and len(main_word) < 15:
                hashtags.append(f"#{main_word}")

        if len(clean_words) > 1:
            second_word = clean_words[1]
            if len(second_word) > 3 and len(second_word) < 15 and second_word not in ["и", "в", "на", "с", "по"]:
                hashtags.append(f"#{second_word}")

        return hashtags[:3]

    @staticmethod
    def _default_prompt() -> str:

        return (
            "Ты — профессиональный редактор русскоязычного Telegram-канала. "
            "Твоя задача — переработать новость в привлекательный пост для Telegram.\n\n"
            "ИНСТРУКЦИИ ПО ФОРМАТИРОВАНИЮ (СТРОГО СЛЕДУЙ ЭТИМ ПРАВИЛАМ):\n"
            "1. Создай ЗАГОЛОВОК: сделай его ЖИРНЫМ (<b>текст</b>), добавь 1 релевантный эмодзи в начало.\n"
            "2. После заголовка добавь ОДИН ПУСТОЙ АБЗАЦ (два символа перевода строки).\n"
            "3. Основной текст: 2-3 информативных абзаца с ОТСТУПАМИ МЕЖДУ НИМИ (два символа перевода строки).\n"
            "4. В конце добавь 2-3 релевантных хештега с ОТСТУПОМ ПЕРЕД НИМИ (один пустой абзац).\n"
            "5. НИКОГДА не добавляй нумерацию, маркеры списка, подзаголовки или дополнительные форматирования.\n"
            "6. НИКОГДА не включай в ответ части этого промпта, инструкции или метаинформацию.\n\n"
            "ПРИМЕР ИДЕАЛЬНОГО ПОСТА:\n"
            "<b>💡 Россиянам напомнили о шестидневной рабочей неделе</b>\n\n"
            "Согласно информации от Федеральной службы по труду и занятости, россияне могут вернуться к шестидневной рабочей неделе. Это решение обусловлено ростом объемов работ и потребностями бизнеса в увеличении производительности.\n\n"
            "В службе отметили, что такая форма рабочей организации может применяться только при согласии работников и соблюдении всех трудовых норм. Также поднимался вопрос о необходимости адаптации законодательства под изменения в трудовых отношениях.\n\n"
            "Сейчас многие компании рассматривают возможность внедрения новых графиков, учитывая мнение сотрудников и состояние рынка.\n\n"
            "#труд #работа #россия"
        )