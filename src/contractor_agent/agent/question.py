"""Shared question predicates. Context words must not select a canned answer."""

import re


def subject(question: str) -> str:
    return re.split(r"Речь о компании|(?:Контекст сравнения|Компании):", question, maxsplit=1)[
        0
    ].strip()


BANK_LABEL = re.compile(
    r"\bзск\b|светофор|\bметк[аиу]\b|оценк[аиу]\s+банка"
    r"|индикатор\w*[^.!?\n]{0,35}(?:зел[её]н|красн|ж[её]лт|сер)"
    r"|(?:банковск\w*|цвет\w*)\s+индикатор",
    re.I,
)
RATING_REQUEST = re.compile(
    r"(?:поставь|присвой|дай|оцени|выставь|оценк|рейтинг|балл)[^.!?\n]{0,55}"
    r"(?:от\s*1\s*до\s*10|1\s*[-–—/]\s*10|десятибалльн|балл|по\s+шкал)"
    r"|(?:rate|score)[^.!?\n]{0,50}(?:1\s*(?:to|-)\s*10|out of 10)",
    re.I,
)
COURT_QUESTION = re.compile(
    r"\bсуд|арбитраж|\bиск(?:и|ов|ах|ам|ами|е|а|у)?\b|истц|истец|ответчик"
    r"|сколько.{0,18}дел|в какой роли",
    re.I,
)
STAFF_QUESTION = re.compile(
    r"численност|\bштат\w*\b|сколько[^.!?\n]{0,30}(?:сотрудник|работник|персонал|работает)"
    r"|(?:есть\s+ли|имеются\s+ли)[^.!?\n]{0,20}(?:сотрудник|работник|персонал)"
    r"|^(?:а\s+)?(?:сотрудники|персонал)[?!.\s]*$",
    re.I,
)
EMAIL_QUESTION = re.compile(r"электронн\w*\s+(?:адрес|почт)|\be-?mail\b|\bимейл\b|\bпочт\w*", re.I)
HEAD_IDENTITY = re.compile(
    r"\b(?:кто|фио|имя)\b[^.!?\n]{0,45}(?:руководител|руководит|директор|управляющ)"
    r"|\b(?:руководител\w*|директор\w*)\b[^.!?\n]{0,25}\b(?:кто|зовут)\b"
    r"|\bкак\s+зовут\b[^.!?\n]{0,25}(?:руководител|директор|управляющ)",
    re.I,
)
COOPERATION = re.compile(
    r"(?:стоит|можно|можем|рекоменду\w*|совету\w*|безопасно|хочу|будем|лучше)"
    r"[^.!?\n]{0,65}(?:работать|сотруднич|сотр[оу]днич|иметь\s+дело|связываться)"
    r"|(?:работать|сотруднич\w*|иметь\s+дело|связываться)[^.!?\n]{0,40}"
    r"(?:стоит|можно|рекоменду\w*|совету\w*|или\s+нет)"
    r"|в\s+поставщики|(?:сотрудничеств\w*|работа\s+с\s+ними)[^.!?\n]{0,30}(?:да\s+или\s+нет|тво[её]\s+мнение)",
    re.I,
)
MEANING = re.compile(
    r"что\s+(?:это\s+)?(?:вообще\s+)?(?:значит|означает)|как\s+(?:это\s+)?понимать"
    r"|поясни|объясни|чем[^.!?\n]{0,30}гроз|на\s+что[^.!?\n]{0,30}(?:повлия|влия)"
    r"|как[^.!?\n]{0,35}(?:повлия|влия|скажется|отразится)|последств|в\s+ч[её]м[^.!?\n]{0,20}опасност",
    re.I,
)
CHOICE = re.compile(
    r"с кем.{0,25}(?:лучше|работать)|кого\s+(?:бы\s+ты\s+)?"
    r"(?:выбр|выбер|выбират|выбра|порекоменд|рекоменд|посовет)"
    r"|какого[^.!?\n]{0,45}(?:выбр|выбер|порекоменд|рекоменд|посовет)"
    r"|кто\s+из[^.!?\n]{0,35}лучше|кому[^.!?\n]{0,25}(?:предпочт|отдать)"
    r"|кого[^.!?\n]{0,50}(?:выбр|выбер|рекоменд|посовет)"
    r"|кому[^.!?\n]{0,35}(?:дать|давать|предоставить)[^.!?\n]{0,15}отсроч",
    re.I,
)
TRANSACTION_DECISION = re.compile(
    r"(?:можно|стоит|рекоменду\w*|совету\w*|безопасно|лучше)[^.!?\n]{0,45}"
    r"(?:дать|давать|предоставить|согласиться|выбрать|отгружать|платить|отсроч|предоплат|постоплат)"
    r"|(?:отсроч\w*|постоплат\w*|предоплат\w*)[^.!?\n]{0,35}(?:или\s+лучше|стоит|можно|или\s+нет)",
    re.I,
)


def recommendation_requested(question: str) -> bool:
    q = subject(question)
    return bool(COOPERATION.search(q) or CHOICE.search(q) or TRANSACTION_DECISION.search(q))


DECISION = re.compile(
    r"\bможно\b[^.!?\n]{0,45}(?:работ|плат|отсроч|подпис|постав|отгруж|заключ)"
    r"|\b(?:стоит|безопасно|рискованно|страшно|надо)\s+ли\b"
    r"|пройд[её]т\s+ли|потянет\s+ли|есть\s+ли\s+смысл|взыщу\s+ли"
    r"|отсроч|постоплат|предоплат|с\s+кем.{0,25}работать|в\s+поставщики"
    r"|(?:собираемся|собираюсь|планируем|хотим|хочу)[^.!?\n]{0,35}(?:подпис|заключ|оплат|взять)"
    r"|(?:руководител|директор)[^.!?\n]{0,30}(?:нормально|порядке|достоверн)"
    r"|(?:оплачу|оплатить|подписывать|заключать|страшно)\b",
    re.I,
)
RISK_TOPIC = re.compile(
    r"риск|над[её]жн|недостовер|банкрот|ликвидир|исключен|исключён|блокиров"
    r"|\bрнп\b|недобросовестн\w*\s+поставщик|репутац|вс[её]\s+(?:ли\s+)?(?:нормально|чисто|в\s+порядке)",
    re.I,
)


def needs_risk_review(question: str) -> bool:
    q = subject(question)
    return bool(
        DECISION.search(q) or COOPERATION.search(q) or CHOICE.search(q) or RISK_TOPIC.search(q)
    )


def needs_interpretation(question: str) -> bool:
    """Interpretation is composed by the model, never replaced by a report template."""
    q = subject(question)
    return (
        not limitation(q)
        and not is_full_review(q)
        and bool(recommendation_requested(q) or MEANING.search(q))
    )


def resolved_status_followup(question: str, inn: str) -> bool:
    """Only a resolved identity or an unambiguously pronominal status question.

    Unknown/new names must still go through search; do not inherit the last company.
    """
    q = subject(question)
    if limitation(q) != "fresh_status":
        return False
    ids = set(re.findall(r"\b(?:\d{10}|\d{12})\b", q))
    if ids:
        return ids == {inn}
    words = set(re.findall(r"[а-яёa-z]+", q.casefold()))
    return bool(words) and words <= {
        "а",
        "и",
        "но",
        "ну",
        "так",
        "то",
        "есть",
        "прямо",
        "сейчас",
        "сегодня",
        "теперь",
        "уже",
        "ещё",
        "еще",
        "на",
        "в",
        "настоящий",
        "момент",
        "ли",
        "какой",
        "какая",
        "какое",
        "что",
        "со",
        "с",
        "у",
        "неё",
        "нее",
        "него",
        "она",
        "он",
        "оно",
        "это",
        "эта",
        "этот",
        "этой",
        "этого",
        "её",
        "ее",
        "его",
        "компания",
        "компании",
        "организация",
        "организации",
        "контрагент",
        "контрагента",
        "действует",
        "действующая",
        "действующий",
        "работает",
        "закрыта",
        "закрыт",
        "закрылась",
        "ликвидирована",
        "ликвидирован",
        "статус",
        "текущий",
        "или",
        "?",
    }


def is_full_review(question: str) -> bool:
    q = subject(question)
    if limitation(q):
        return False
    if re.search(r"что\s+(?:нужно\s+)?проверить|какие\s+проверки", q, re.I):
        return False
    if re.search(r"проверь\s+(?:только\s+)?(?:метк|оценк|светофор|зск)", q, re.I):
        return False
    if not re.search(
        r"\bпроверь\b|\bпроверить\b|полная проверка|полн\w*\s+карточк|"
        r"что\s+(?:можешь\s+)?сказать\s+(?:о|об|про)|расскажи.{0,15}компан",
        q,
        re.I,
    ):
        return False
    return not re.search(
        r"лиценз|телефон|численност|\bштат|выручк|прибыл|ликвидн|\bсуды\b|"
        r"арбитраж|пристав|руководител|директор|учредител|адрес|выписк|\bрнп\b|почт|email|сайт|возраст|статус",
        q,
        re.I,
    )


def limitation(question: str) -> str | None:
    q = subject(question)
    if re.search(
        r"(?:оплатил[аи]?|оплачен|уплачен|перев[её]л|отправил[аи]?)[^.!?\n]{0,45}(?:ли\b|сч[её]т|деньг|плат[её]ж)"
        r"|(?:проверь|узнай|покажи)[^.!?\n]{0,55}(?:мо[ийю]\s+(?:плат[её]ж|перевод)|операци\w*\s+по\s+сч[её]т)",
        q,
        re.I,
    ):
        return "payment_execution"
    if re.search(r"зайди|прочитай|посмотри|поищи|найди|проверь", q, re.I) and re.search(
        r"(?:на\s+|веб[- ]?)сайт|в интернет|онлайн", q, re.I
    ):
        return "external_data"
    if re.search(r"финанс|выручк|прибыл|оборот", q, re.I) and re.search(
        r"(?:за|по)\s+(?:[а-яё0-9]+\s+){0,2}(?:месяц|квартал)|помесячн|поквартальн"
        r"|(?:месячн|квартальн)\w*\s+(?:выручк|прибыл|финанс|отч[её]т)",
        q,
        re.I,
    ):
        return "financial_period"
    if re.search(r"плат[её]ж|перевод|перев[её]л|деньг", q, re.I) and re.search(
        r"дошл|дойд|зачисл|поступил|уже получил|прош[её]л|прошли|пройд", q, re.I
    ):
        return "payment_execution"
    # Current court cases, proceedings or account restrictions are not the
    # company's registration status. Match the clause, not unrelated words
    # anywhere in a long question about a company.
    for clause in re.split(r"[.!?;\n]", q):
        if (
            re.search(r"сейчас|сегодня|в\s+настоящий\s+момент|текущ", clause, re.I)
            and re.search(
                r"статус|действует|действующ|закрыт|ликвидиров|банкрот|исключен|исключён",
                clause,
                re.I,
            )
            and not re.search(r"выписк", clause, re.I)
            and not re.search(
                r"\bдел\w*\b|\bсуд\w*\b|арбитраж|\bиск\w*\b|производств|пристав|сч[её]т|лиценз",
                clause,
                re.I,
            )
        ):
            return "fresh_status"
    if re.search(
        r"на сегодня|на текущ\w*\s+дат|ничего не поменялось|что изменилось|"
        r"свеж\w*\s+(?:данн|сведени|отч[её]т)|актуальн\w*\s+(?:статус|сведени|данн)",
        q,
        re.I,
    ) and not re.search(r"выписк", q, re.I):
        return "fresh_status"
    if re.search(r"выписк", q, re.I) and re.search(
        r"свеж|актуальн|новую|получ|пришли|выдай|скачай|сформируй|закажи|сделай", q, re.I
    ):
        return "fresh_extract"
    if re.search(
        r"(?:платят|платит|оплачивают|оплачивает|рассчитыва\w*|перечисля\w*|перевод\w*)[^.!?\n]{0,65}"
        r"(?:вовремя|своевременно|в\s+срок)|плат[её]жн\w*\s+дисциплин",
        q,
        re.I,
    ):
        return "payment_history"
    if re.search(
        r"(?:плат[её]ж|перевод|деньги)[^.!?\n]{0,55}(?:дойд|дош|пройд|прош|получ)"
        r"|(?:пройд|дойд)[^.!?\n]{0,30}(?:плат[её]ж|перевод|деньги)",
        q,
        re.I,
    ):
        return "payment_execution"
    if re.search(r"адрес\s+проживания|домашн\w*\s+адрес|где\s+(?:он[аи]?\s+)?жив[её]т", q, re.I):
        return "home_address"
    return None
