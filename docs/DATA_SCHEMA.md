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

Словари признаков тоже не идентичны: код `dishonestProvider` встречается только
в CSV-наборе.

**Вывод: работать надо с обоими, объединяя. Не считать CSV «плоской версией
JSON» — это другие компании.**

## Даты

Файл назван «снапшот на 27.08.2026», но это дата выгрузки. Отчёты внутри
сформированы **с 29.07 по 27.08.2026, 25 разных дат** в каждом наборе.
У каждой компании своя дата отчёта, поле `report.reportDate`.

---

## JSON: структура записи

Корень — массив из 100 объектов. Каждый объект:

```json
{
  "_id":    { "date": {...}, "ogrn": "1241600001048" },
  "report": { ... }            // всё содержательное здесь
}
```

Внутри `report` — 18 возможных секций. **Схема плавающая: набор секций
отличается от компании к компании.**

### Есть у всех 100

| Секция | Тип | Что внутри |
|---|---|---|
| `reportDate` | `{"$date": ISO}` | когда сформирован отчёт |
| `baseInfo` | объект | ИНН, ОГРН, ОКПО, названия, адрес, КПП, дата регистрации, `riskLevel`, размер, email, сайт |
| `kindsOfActivityInfo` | объект | основной ОКВЭД + список дополнительных (до 120 штук) |
| `status` | объект | `status` (у всех `CURRENT`), `date`, **`reasonName`** |
| `zskRiskLevel` | строка | вторая метка риска: `GREEN` / `YELLOW` / `RED` |
| `reputationalRisks` | объект | `negative[]` и `positive[]` — готовые тексты с рекомендациями |
| `phones` | список | часто пустой |
| `arbitrationByStatus` | объект | агрегат по арбитражу; ключ есть всегда, содержимое часто пустое |
| `executionProceedings` | список | исполнительные производства; часто пустой |
| `procurements` | список | госзакупки; часто пустой |

### Есть не у всех — заполненность JSON / CSV из 100

| Секция | JSON | CSV | Что внутри |
|---|---:|---:|---|
| `foundersInfo` | 75 | 75 | уставный капитал, учредители с долями, руководитель |
| `taxSystem` | 75 | 75 | режим налогообложения |
| `finReports` | 67 | 69 | отчётность до 3 лет: активы, пассивы, выручка, прибыль |
| `relatedCompanies` | 61 | 65 | связанные компании по ИНН/ОГРН |
| `arbitrationCases` | 44 | 57 | **не список дел, а разбивка по годам** |
| `inspections` | 30 | 29 | проверки надзорных органов |
| `coefficient` | 19 | 28 | рентабельность, платёжеспособность, устойчивость |
| `licenses` | 9 | 13 | лицензии |
| `branchesInfo` | 2 | 6 | филиалы |

### Заполненность внутри `baseInfo` (из 100, JSON)

`inn`, `ogrn`, `okpo`, `shortName`, `fullName`, `registrationInfo`, `riskLevel` — **100** ·
`kpp` и `address` — **75** · `companySize` — **59** · `email` — **35** · `website` — **11**.

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
├── relatedCompanies[] { inn, ogrn, name, authPersonName, authPersonPosition, registrationDate.$date }
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
| `executionProceedings` | **1693** | `[0..458]` — 459 элементов × 4 поля |
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
| остальное | 11 | `coefficient`, `status`, `_id`, `taxSystem`, `reportDate`, `zskRiskLevel` |

### Что это значит на практике

**Ширина таблицы определяется самой «толстой» компанией набора.** В CSV-наборе
у кого-то 459 исполнительных производств — и колонки под них заведены для всех
ста строк. У компании с двумя производствами 457 × 4 колонки пустые.
**Заполненность CSV — единицы процентов.**

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
объединённым набором на 200 записей.** Конвертация тривиальна: разбираем имена
колонок по точкам и индексам в скобках, склеиваем обратно в дерево, выбрасываем
пустые.

---

## Мины при парсинге — обязательно к прочтению

**1. Числа приходят то числом, то объектом.** MongoDB extended JSON:
```json
"commonAmount": 957189
"commonAmount": { "$numberLong": "4534783044" }
```
Затронуто 12 полей: `arbitrationByStatus.commonAmount`, `dfAmount`,
`finReports[].assets.totalAssets`, `.currentAssets.receivables`, `.stocks`,
`.total`, `.uncurrentAssets.total`, `common.proceeds`,
`liabilities.shortTermLiabilities.total`, `.accountsPayable`,
`.longTermDuties.total`, `totalLiabilities`, `foundersInfo.shareCapital`,
`cofounders[].amount`. Встречается у 1–6 записей на поле — то есть код упадёт
не сразу, а на середине выборки.

**2. Часть чисел — строки.**
`executionProceedings[].amount` = `"517235.54"` — строка, и заполнена только
у 36 записей из 53. `coefficient.profitability / solvency / sustainability`
= `"0.5"`, `"1.04"`.

**3. Две несовместимые схемы дат в одном документе.**
```json
"registrationDate": { "$date": "2024-01-14T21:00:00.000Z" }   // объект
"startDate": "2023-11-17"                                      // строка
```
`$date` — везде, кроме `inspections`, где даты плоскими строками.
И `21:00:00Z` — это московская полночь: при наивном парсинге в UTC дата
съезжает на день назад.

**4. `arbitrationCases` — не дела, а годовая разбивка.**
`{year, plaintiffCount, plaintiffAmount, defendantCount, defendantAmount}`.
Отдельных дел с номерами в данных нет вообще. Плюс агрегат
`arbitrationByStatus` и разбивка расходятся: агрегат непустой у 48 компаний,
разбивка — у 44; у шести есть агрегат без разбивки, у двух наоборот.

**5. Один код — оба знака.** В `reputationalRisks` 11 из 12 негативных кодов
встречаются и в `positive`: `fnsBlocking` в позитивных значит «блокировок нет»,
в негативных — «есть». Знак определяется секцией, а не кодом.

**6. Объём.** Карточка «ЛЕ МОНЛИД» с 1744 производствами и 1525 арбитражными
делами не влезет в контекст модели. Нужна агрегация до подачи: свернуть
в «активных N на сумму X, завершённых M», полный список отдавать только
по явному запросу.
