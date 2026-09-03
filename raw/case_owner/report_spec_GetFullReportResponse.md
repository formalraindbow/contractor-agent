# Спецификация отчёта — GetFullReportResponse

От кейсодателя, получено 3.09. Описание полей отчёта системы проверки
контрагента — того самого формата, в котором лежат оба снапшота в `data/`.
Разбор семантики меток и перечислений — `docs/DATA_SCHEMA.md`.

## Общая информация

| Поле | Значение |
|---|---|
| `reportDate` | Дата формирования отчёта |
| `baseInfo.inn` / `ogrn` / `kpp` / `okpo` | ИНН, ОГРН, КПП, ОКПО |
| `baseInfo.shortName` / `fullName` | Краткое и полное наименование |
| `baseInfo.riskLevel` | Уровень риска контрагента: LOW / MEDIUM / HIGH |
| `baseInfo.address` / `email` / `website` | Юридический адрес, e-mail, сайт |
| `baseInfo.companySize` | Размер организации (например, «Микропредприятие») |
| `baseInfo.staff` | Диапазон численности персонала |
| `baseInfo.registrationInfo.registrationDate` / `yearsFromRegistration` | Дата регистрации, сколько лет существует |
| `phones[].phoneType` / `phoneCode` / `phoneNumber` | Телефоны |
| `status.status` | CURRENT (действующая) / CLOSED (ликвидированная) |
| `status.reasonName` | Причина закрытия организации |
| `status.date` | Дата последнего обновления статуса |
| `zskRiskLevel` | Уровень риска «Знай своего клиента»: GREEN / YELLOW / RED (в интерфейс выводится Green / grey / grey) |

## Учредители, руководство, структура

| Поле | Значение |
|---|---|
| `foundersInfo.shareCapital` | Уставной капитал |
| `foundersInfo.cofounders[].name` / `inn` / `amount` / `share` / `dateFrom` / `isActive` | Учредитель: ФИО, ИНН, сумма доли, доля в %, дата вхождения, активен ли |
| `foundersInfo.authPerson.name` / `positionName` / `inn` / `positionDate` | Руководитель: ФИО, должность, ИНН, дата вступления |
| `foundersInfo.parentOrganizations[].inn` / `ogrn` / `fullName` / `parentDate` | Управляющая компания |
| `relatedCompanies[].inn` / `ogrn` / `name` / `registrationDate` / `authPersonName` / `authPersonPosition` / `parentOrganizations[]` | Связанная организация |
| `kindsOfActivityInfo.mainKindOfActivity.code` / `description` | Основной ОКВЭД |
| `kindsOfActivityInfo.otherKindsOfActivity[].code` / `description` | Дополнительные ОКВЭД |
| `branchesInfo.branchesCount` / `branches[].name` / `address` | Филиалы |
| `taxSystem[].fullName` / `shortName` | Режим налогообложения |

## Юридические риски

| Поле | Значение |
|---|---|
| `arbitrationCases[].year` / `plaintiffCount` / `plaintiffAmount` / `defendantCount` / `defendantAmount` | По годам: число и сумма дел как истец и как ответчик |
| `arbitrationByStatus.commonCount` / `commonAmount` | Общее количество и сумма арбитражных дел |
| `arbitrationByStatus.plaintiffArbitration.plaintiffArbitrationFinished.pfCount` / `pfAmount` | Закрытые дела как истец |
| `…plaintiffArbitrationAppealed.paCount` / `paAmount` | Обжалованные дела как истец |
| `…plaintiffArbitrationPending.ppCount` / `ppAmount` | Открытые дела как истец |
| `arbitrationByStatus.defandantArbitration.defandantArbitrationFinished.dfCount` / `dfAmount` | Закрытые дела как ответчик |
| `…defandantArbitrationAppealed.daCount` / `daAmount` | Обжалованные дела как ответчик |
| `…defandantArbitrationPending.dpCount` / `dpAmount` | Открытые дела как ответчик |
| `executionProceedings[].active` / `number` / `date` / `amount` | Исполнительное производство: активно ли, номер, дата, сумма |
| `inspections[].erpId` / `type` / `form` / `authorityName` / `startDate` / `endDate` / `inspectionStatus` | Проверка; статус: предстоящая / завершена без нарушений / с нарушениями / результат неизвестен / отменена |
| `licenses[].number` / `name` / `issuingAuthority` / `issueDate` / `endDate` / `status` | Лицензия; статус: ACTIVE / EXPIRED / INDEFINITE |

## Репутационные риски

| Поле | Значение |
|---|---|
| `reputationalRisks.negative[].code` / `name` / `chapter` | Негативный фактор: код, описание, раздел |
| `reputationalRisks.positive[].code` / `name` / `chapter` | Позитивный фактор: код, описание, раздел |

## Финансовые показатели

| Поле | Значение |
|---|---|
| `finReports[].common.year` / `proceeds` / `profit` | Год, выручка, прибыль (убыток) |
| `finReports[].assets.totalAssets` | Все активы |
| `finReports[].assets.currentAssets.total` / `stocks` / `receivables` / `bankroll` | Оборотные активы: всего, запасы, дебиторская задолженность, денежные средства |
| `finReports[].assets.uncurrentAssets.total` / `fixedAssets` | Внеоборотные активы: всего, основные средства |
| `finReports[].liabilities.totalLiabilities` / `capitals` | Пассивы всего, капиталы и резервы |
| `finReports[].liabilities.longTermDuties.total` / `others` | Долгосрочные обязательства |
| `finReports[].liabilities.shortTermLiabilities.total` / `borrowedFunds` / `accountsPayable` | Краткосрочные обязательства: всего, заёмные, кредиторская задолженность |
| `coefficient.year` / `sustainability` / `solvency` / `profitability` | Коэффициенты: устойчивость, платёжеспособность, рентабельность |

## Госзакупки

| Поле | Значение |
|---|---|
| `procurements[].tenderAdmittedCnt` / `tenderWinnerCnt` / `contractSignedCnt` / `contractSignedAmt` / `federalLawCode` / `procurementsYear` | Тендеры: допущен, выиграл, контрактов подписано и на какую сумму, закон, год |

Ссылка, присланная вместе со спецификацией:
https://www.cbr.ru/counteraction_m_ter/platform_zsk/proverka-po-inn/
