# Структура данных — JSON и CSV

Техническое описание выгрузки из Drive-папки хакатона. Что внутри, чем файлы
отличаются, с чем работать. Находки и ловушки по содержанию — в `DATA_ANALYSIS.md`.

---

## Два файла — и это два разных набора данных

| | `2026-08-27_снапшот_100_контрагентов.json` | `2026-08-27_снапшот_100_контрагентов.csv` |
|---|---|---|
| размер | 2,5 МБ | 1,5 МБ |
| записей | 100 | 100 |
| формат | вложенный JSON, массив объектов | плоская таблица, **2654 колонки** |

🔴 **Это не один набор в двух форматах.** Пересечение по `inn` и по `_id.ogrn` —
**ноль записей**. Итого 200 уникальных компаний. Совпадает одно короткое
название (ООО «ОМЕГА»), но ИНН разные.

Словари признаков тоже не идентичны: негативные коды `dishonestProvider`,
`taxArrears` и `inspectionWithViolation` встречаются только в CSV-наборе (по одной
компании), а негативные `currentAssets`, `liquidationStatus`, `massAuthpersons` —
только в JSON. Позитивный `dishonestProvider` есть в обоих у всех. Итого негативных
кодов в объединении 15 — словарь открытый.

**Вывод: работать надо с обоими, объединяя. Не считать CSV «плоской версией
JSON» — это другие компании.**

## Даты

Файл назван «снапшот на 27.08.2026», но это дата выгрузки. Отчёты внутри
сформированы **с 30.07 по 28.08.2026 по Москве, 25 разных дат** в каждом наборе
(28 в объединении). У каждой компании своя дата отчёта, поле `report.reportDate`.
В строке `$date` лежит московская полночь (`…T21:00:00.000Z`), поэтому UTC-часть
строки на день раньше настоящей даты — см. мину 3.

---

## JSON: структура записи

Корень — массив из 100 объектов. Каждый объект:

```json
{
  "_id":    { "date": {...}, "ogrn": "1241600001048" },
  "report": { ... }            // всё содержательное здесь
}
```

`_id` избыточен: `_id.date` равен `report.reportDate`, а `_id.ogrn` —
`report.baseInfo.ogrn` во всех 200 записях. Ключ записи — ИНН (строка, уникален
в обоих файлах).

Внутри `report` — 19 возможных ключей: 18 секций и `reportDate`. **Схема плавающая:
набор секций отличается от компании к компании.**

### Есть у всех 100

| Секция | Тип | Что внутри |
|---|---|---|
| `reportDate` | `{"$date": ISO}` | когда сформирован отчёт |
| `baseInfo` | объект | ИНН, ОГРН, ОКПО, названия, адрес, КПП, дата регистрации, `riskLevel`, размер, email, сайт |
| `kindsOfActivityInfo` | объект | основной ОКВЭД + список дополнительных (до 120 штук) |
| `status` | объект | `status` (у всех `CURRENT`), `date`, **`reasonName`** |
| `zskRiskLevel` | строка | вторая метка риска: `GREEN` / `YELLOW` / `RED` |
| `reputationalRisks` | объект | `negative[]` и `positive[]` — готовые тексты с рекомендациями; `negative` пуст у 42 |
| `phones` | список | у 71 из 100 — `[]` |
| `arbitrationByStatus` | объект | агрегат по арбитражу; ключ есть всегда, но у 52 нет `commonCount`/`commonAmount` — только пустые вложенные словари |
| `executionProceedings` | список | исполнительные производства; у 47 — `[]` (ключ есть всегда) |
| `procurements` | список | госзакупки; у 92 — `[]` |

### Есть не у всех — заполненность JSON / CSV из 100

| Секция | JSON | CSV | Что внутри |
|---|---:|---:|---|
| `foundersInfo` | 75 | 75 | уставный капитал, учредители с долями, руководитель |
| `taxSystem` | 75 | 75 | режим налогообложения |
| `finReports` | 67 | 69 | отчётность до 3 лет: активы, пассивы, выручка, прибыль; в JSON ключ есть у 75, у 8 из них `[]` (в т.ч. ЛЕ МОНЛИД) |
| `relatedCompanies` | 61 | 65 | связанные компании по ИНН/ОГРН |
| `arbitrationCases` | 44 | 57 | **не список дел, а разбивка по годам** |
| `inspections` | 30 | 29 | проверки надзорных органов |
| `coefficient` | 19 | 28 | рентабельность, платёжеспособность, устойчивость |
| `licenses` | 9 | 13 | лицензии |
| `branchesInfo` | 2 | 6 | филиалы |

### Заполненность внутри `baseInfo` (из 100, JSON)

`inn`, `ogrn`, `okpo`, `shortName`, `fullName`, `registrationInfo`, `riskLevel` — **100** ·
`kpp` — **75** (у 25 ИП ключа нет) · `address` — ключ у всех 100, у 25 ИП пустая строка · `companySize` — **59** · `email` — **35** · `website` — **11**.

### Пример дерева одной записи

```
report
├── reportDate.$date                       "2026-08-27T21:00:00.000Z"
├── baseInfo
│   ├── inn / ogrn / okpo / shortName / fullName
│   ├── registrationInfo.registrationDate.$date
│   ├── registrationInfo.yearsFromRegistration     int
│   ├── riskLevel                                  LOW | MEDIUM | HIGH | UNKNOWN
│   ├── address / kpp / companySize / email / website
├── status
│   ├── status         CURRENT
│   ├── date.$date
│   └── reasonName     ← банкротство, исключение из ЕГРЮЛ (заполнено у 6 из 200)
├── zskRiskLevel                                   GREEN | YELLOW | RED
├── reputationalRisks
│   ├── negative[] { code, name, chapter }
│   └── positive[] { code, name, chapter }
├── kindsOfActivityInfo
│   ├── mainKindOfActivity { code, description }
│   └── otherKindsOfActivity[] { code, description }
├── foundersInfo
│   ├── shareCapital
│   ├── authPerson { inn, name, positionName, positionDate.$date }
│   └── cofounders[] { inn, name, amount, share, dateFrom.$date, active }
├── finReports[]                                   до 3 лет
│   ├── common { year, proceeds, profit }
│   ├── assets { totalAssets, currentAssets{...}, uncurrentAssets{...} }
│   └── liabilities { capitals, shortTermLiabilities{...}, longTermDuties{...} }
├── coefficient { year, profitability, solvency, sustainability }   строки!
├── arbitrationByStatus
│   ├── commonCount / commonAmount
│   ├── plaintiffArbitration { ...Finished{pfCount,pfAmount}, ...Pending, ...Appealed }
│   └── defandantArbitration { ...Finished{dfCount,dfAmount}, ...Pending, ...Appealed }
├── arbitrationCases[] { year, plaintiffCount, plaintiffAmount, defendantCount, defendantAmount }
├── executionProceedings[] { number, date.$date, amount, active }
├── inspections[] { authorityName, form, inspectionStatus, startDate, endDate, erpId, type }
├── licenses[] { number, name, issuingAuthority, status, issueDate.$date, endDate.$date }
├── relatedCompanies[] { inn, ogrn, name, authPersonName?, authPersonPosition?, registrationDate.$date?,
│                        parentOrganizations[]? { inn, ogrn, fullName, parentDate.$date } }
├── branchesInfo { branchesCount, branches[] { name, address } }
├── procurements[] { procurementsYear, federalLawCode, tenderWinnerCnt, contractSignedCnt, contractSignedAmt }
├── taxSystem[] { shortName, fullName }
└── phones[] { phoneCode, phoneNumber }
```

Обрати внимание на опечатку в исходной схеме: **`defandantArbitration`**, а не
`defendant`. В `arbitrationCases` при этом правильное `defendantCount`.
В коде легко промахнуться.

---

## CSV: та же схема, сплющенная в 2654 колонки

Вложенность склеена точками, элементы списков развёрнуты в отдельные колонки
с индексом:

```
report.baseInfo.inn
report.baseInfo.registrationInfo.registrationDate
report.reputationalRisks.negative[0].code
report.reputationalRisks.negative[1].code
report.executionProceedings[0].number
report.executionProceedings[458].number
report.relatedCompanies[2].parentOrganizations[0]
```

### Сколько колонок на что

| Секция | Колонок | До какого индекса развёрнут список |
|---|---:|---|
| `executionProceedings` | **1693** | `[0..458]` — 459 × 3 поля (`number`, `date`, `active`) + 316 колонок `amount` |
| `inspections` | 335 | `[0..47]` |
| `kindsOfActivityInfo` | 180 | `otherKindsOfActivity[0..88]` |
| `relatedCompanies` | 103 | `[0..16]` |
| `reputationalRisks` | 78 | `positive[0..20]`, `negative[0..4]` |
| `finReports` | 51 | `[0..2]` |
| `licenses` | 50 | `[0..8]` |
| `phones` | 30 | `[0..14]` |
| `foundersInfo` | 28 | `cofounders[0..3]` |
| `procurements` | 25 | `[0..4]` |
| `branchesInfo` | 21 | `branches[0..11]` |
| `arbitrationCases` | 20 | `[0..3]` |
| `arbitrationByStatus` | 14 | — |
| `baseInfo` | 13 | — |
| остальное | 13 | `coefficient` 4, `status` 3, `_id` 2, `taxSystem` 2, `reportDate`, `zskRiskLevel` |

### Что это значит на практике

**Ширина таблицы определяется самой «толстой» компанией набора.** В CSV-наборе
у кого-то 459 исполнительных производств — и колонки под них заведены для всех
ста строк. У компании с двумя производствами 457 × 4 колонки пустые.
**Заполненность CSV — единицы процентов.**

**Заголовок — разреженное объединение путей, а не плотная сетка.** Колонка
заводится, если поле непустое хотя бы у одной компании: `amount` есть у 316
из 459 индексов производств, `licenses[].endDate` — только у [0], [1], [6], [7], [8].
Порядок колонок — «первое появление» по строкам, группами по полю, а не по
элементу (`cofounders[0].name, [1].name, …, потом [0].inn`), индексы внутри группы
не обязательно по возрастанию. Разбирать можно только по имени колонки, никогда
по позиции. Внутри одной строки индексы заполненных элементов сплошные от 0.

**Обёрток Mongo в CSV нет.** `$date` стал голой строкой `2024-01-14T21:00:00.000Z`,
`$numberLong` — цифрами, `true`/`false` — строками, пусто — `''`. Все ячейки —
строки, типизировать их надо по пути поля, а не по виду значения: ОКВЭД `31.0`,
ИНН `0277985654`, `erpId` из 20 цифр численный разбор испортит.

Для сравнения, в JSON-наборе максимум ещё больше: **1744 исполнительных
производства** у ООО «ЛЕ МОНЛИД», 120 дополнительных ОКВЭД, 100 проверок,
34 связанные компании. Если бы этот набор сплющивали так же, колонок было бы
около семи тысяч.

---

## Чем пользоваться

**JSON — рабочий формат.** Вложенность соответствует смыслу, списки нормальной
длины, парсится в один `json.load`, отсутствующие секции просто отсутствуют.

**CSV — только для быстрого взгляда в Excel** и для проверки гипотез по плоским
полям (`riskLevel`, `zskRiskLevel`, ИНН, названия). Для кода агента он плох:
чтобы собрать список производств компании, придётся пройти 459 групп колонок
и отфильтровать пустые.

**Но данные в CSV нужны** — там сто компаний, которых нет в JSON. Правильный ход:
**сконвертировать CSV обратно в JSON-подобную структуру и работать с одним
объединённым набором на 200 записей.** Разбор имён колонок прост: точки и индексы
в скобках → дерево, пустые ячейки выбрасываем. Нетривиальны две вещи: типизация
по пути (все ячейки — строки) и политика присутствия — CSV не отличает «секции нет»
от «секция пуста», поэтому секции, которые источник отдаёт всегда
(`executionProceedings`, `phones`, `procurements`, `arbitrationByStatus`,
`reputationalRisks`), при пустых ячейках считаем пустыми, остальные — отсутствующими.

---

## Мины при парсинге — обязательно к прочтению

**1. Числа приходят то числом, то объектом.** MongoDB extended JSON:
```json
"commonAmount": 957189
"commonAmount": { "$numberLong": "4534783044" }
```
Затронуто 16 путей: `arbitrationByStatus.commonAmount`, `dfAmount`,
`arbitrationCases[].defendantAmount`, `finReports[].assets.totalAssets`,
`.currentAssets.receivables`, `.stocks`, `.total`, `.uncurrentAssets.total`,
`common.proceeds`, `liabilities.capitals`, `liabilities.shortTermLiabilities.total`,
`.accountsPayable`, `.longTermDuties.total`, `totalLiabilities`,
`foundersInfo.shareCapital`, `cofounders[].amount`. Правило одно: обёртка у любого
целого ≥ 2³¹ (ни одного «голого» int выше порога), до 15 вхождений на поле
(`common.proceeds`) — код упадёт не сразу, а на середине выборки. Разворачивать
по всему дереву, не по списку полей: на следующем снапшоте список поплывёт.
В CSV обёрток нет.

**2. Часть чисел — строки.**
`executionProceedings[].amount` = `"517235.54"` — строка; ключа нет у 770 из 3873
производств (в том числе у 66 активных из 213; у ЛЕ МОНЛИД — у 33 из 45 активных),
пустой строкой не бывает. Отсутствие суммы — не ноль. «36 из 53» из старой версии —
это сумма только у `[0]`. `coefficient.profitability / solvency / sustainability`
= `"0.5"`, `"1.04"`, бывают отрицательные.

**3. Две несовместимые схемы дат в одном документе.**
```json
"registrationDate": { "$date": "2024-01-14T21:00:00.000Z" }   // объект
"startDate": "2023-11-17"                                      // строка
```
`$date` — везде, кроме `inspections`, где даты плоскими строками `YYYY-MM-DD`.
Все 6703 значения `$date` в обоих файлах — московская полночь, но время в строке
разное: `21:00:00Z` у современных дат и `20:00:00Z` у 198 дат 1993–2014 годов
(UTC+4 и летнее время). Правильный разбор — `zoneinfo("Europe/Moscow")` и `.date()`:
наивная UTC-дата съезжает на день назад у всех, «+3 часа» — у старых. Что настоящая
дата именно московская, видно по календарю: при UTC-трактовке на выходные попадает
1392 даты из 6603 (регистрации, производства приставов), при московской — 142.

**4. `arbitrationCases` — не дела, а годовая разбивка.**
`{year, plaintiffCount, plaintiffAmount, defendantCount, defendantAmount}`.
Отдельных дел с номерами в данных нет вообще. Плюс агрегат
`arbitrationByStatus` и разбивка расходятся: в JSON агрегат непустой у 48,
разбивка — у 44 (6 и 2 в разные стороны), в CSV — 61 и 57 (4 и 0); там, где есть
оба, `commonCount` не равен сумме по годам у 60 компаний из 90, а сумме шести
вложенных счётчиков — у 22. Агрегат нельзя пересчитать из его же частей.

**5. Один код — оба знака.** В `reputationalRisks` после нормализации кириллической
«а» все 12 негативных кодов JSON встречаются и в `positive` (в CSV — 11 из 12,
у `inspectionWithViolation` пары нет): `fnsBlocking` в позитивных значит «блокировок нет»,
в негативных — «есть». Знак определяется секцией, а не кодом.

**6. Объём.** Карточка «ЛЕ МОНЛИД» с 1744 производствами и 1525 арбитражными
делами не влезет в контекст модели. Нужна агрегация до подачи: свернуть
в «активных N на сумму X, завершённых M», полный список отдавать только
по явному запросу.

---

## Дополнения (Алан, 2.09 вечер)

**7. Один код написан кириллицей.** В `reputationalRisks.positive` код `аrbitrationDefendant` (68 вхождений в JSON, 65 в CSV) начинается с **кириллической «а»** (U+0430), в `negative` тот же код — латиницей (`arbitrationDefendant`, 32). Матчинг по коду без нормализации разъезжается; нормализуем при загрузке (явная замена первого символа).

**8. Флаги реестров ФНС существуют только внутри `reputationalRisks`.** `fnsBlocking`, `massAddress`, `invalidAddress`, `invalidRegistrationData`, `massAuthpersons`, `liquidationStatus`, `disqualifiedAuthpersons`, `taxArrears`, `taxReporting`, `dishonestProvider` — отдельных сырых полей под них в отчёте нет. По конспекту QA, в реальных отчётах секция есть у меньшинства — в транскрипте этой фразы нет, не подтверждено; требование работать без секции остаётся. Следствие: без секции агент говорит «сведений о реестрах ФНС в отчёте нет», а не «нарушений нет»; в эталон — те же компании с вырезанной секцией.

**9. Размер карточки.** Медиана — 9,9 тыс. символов (~3–4 тыс. токенов), максимум — 238 тыс. (ЛЕ МОНЛИД); в CSV-наборе максимум 68 тыс. (ЭНЕРГОСПЕЦМОНТАЖ, 459 производств). Даже там, где карточка «влезает», в промпт целиком её не кладём: инструменты отдают агрегаты и нужные секции.

**10. Null в JSON нет ни одного.** Пропуск — всегда отсутствующий ключ, поэтому в модели `X | None = None`. Единственная пустая строка — `baseInfo.address` у 25 ИП (ключ есть у всех 100). Подполя опциональны поодиночке: `*Count` бывает без `*Amount` (в `arbitrationByStatus` и `procurements`), у `inspections` нет `endDate` у 120 из 343 и `type` у 269, у `licenses` `endDate` — у 2 из 14, `otherKindsOfActivity` отсутствует у 3. 42 адреса и часть названий — с концевыми и двойными пробелами в обоих файлах: это данные, а не артефакт CSV. Float в JSON нет вообще, `cofounders[].share` — целые проценты (сумма не всегда 100).

**11. Незадокументированное и мелкое.** `relatedCompanies[].parentOrganizations[] {inn, ogrn, fullName, parentDate.$date}` — вложенный список (JSON 5 элементов, CSV 1). ИНН связанных компаний внутри наших 200 почти не встречаются (2 из 247 в JSON, 0 из 173 в CSV) — это не внешний ключ. Два ООО «ЛЗСО» в JSON (7805327192, 4720028039) — поиск по названию возвращает список. `taxSystem` — всегда ровно один элемент. ИНН с ведущим нулём есть в обоих файлах (`0278949271`, `052500690823` в JSON; `0277985654`, `0232016850` в CSV) — ИНН только строкой. Проверено 2.09 вечером скриптами по обоим файлам.
