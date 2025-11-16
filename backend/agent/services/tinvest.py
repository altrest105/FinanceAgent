import logging
from datetime import datetime, timedelta
from tinkoff.invest import (
    AsyncClient, 
    CandleInterval,
    Quotation, 
    InstrumentIdType,
    PortfolioResponse,
    OperationsResponse,
    GetAccountsResponse,
    OrderDirection,
    OrderType
)
import asyncio
from django.conf import settings

logger = logging.getLogger(__name__)


# ============================================================================
# КОНСТАНТЫ ТИПОВ ОПЕРАЦИЙ
# ============================================================================

class OperationType:
    """Типы операций в Tinkoff Invest API"""
    INPUT = '1'
    OUTPUT = '9'
    BROKER_COMMISSION = '19'
    EXCHANGE_COMMISSION = '20'
    SERVICE_COMMISSION = '18'
    MARGIN_COMMISSION = '17'
    BUY = '10'
    SELL = '11'
    BUY_CARD = '13'
    DIVIDEND = '21'
    TAX_DIVIDEND = '8'
    COUPON = '23'
    TAX_COUPON = '2'
    TAX = '5'
    TAX_LUCRE = '6'
    BENEFIT_TAX = '7'


# ============================================================================
# УТИЛИТЫ ПРЕОБРАЗОВАНИЯ ДАННЫХ
# ============================================================================

def money_value_to_float(money_value):
    """
    Конвертация MoneyValue в float
    
    Args:
        money_value: Объект MoneyValue из Tinkoff API
        
    Returns:
        float: Денежное значение в виде числа
    """
    if not money_value:
        return 0.0
    
    result = money_value.units + money_value.nano / 1e9
    return result


def quotation_to_float(quotation):
    """
    Конвертация Quotation в float
    
    Args:
        quotation: Объект Quotation из Tinkoff API
        
    Returns:
        float: Значение котировки в виде числа
    """
    if not quotation:
        return 0.0
    
    result = quotation.units + quotation.nano / 1e9
    return result


def float_to_quotation(value):
    """
    Конвертация float в Quotation
    
    Args:
        value: Числовое значение
        
    Returns:
        Quotation: Объект для Tinkoff API
    """
    return Quotation(
        units=int(value),
        nano=int((value - int(value)) * 1e9)
    )


def format_currency(value, currency = "RUB", precision = 2):
    """
    Форматирование денежных значений
    
    Args:
        value: Числовое значение
        currency: Код валюты
        precision: Количество знаков после запятой
        
    Returns:
        str: Отформатированная строка с символом валюты
    """
    if value is None:
        return "- ₽"
    
    currency_symbols = {
        "RUB": "₽",
        "USD": "$",
        "EUR": "€",
        "CNY": "¥",
        "GBP": "£"
    }
    
    symbol = currency_symbols.get(currency.upper(), currency)
    formatted_value = f"{value:,.{precision}f}".replace(",", " ")
    
    return f"{formatted_value} {symbol}"


# ============================================================================
# КЭШ ДЛЯ ИНФОРМАЦИИ ОБ ИНСТРУМЕНТАХ
# ============================================================================

_instrument_cache = {}


async def get_instrument_info(client, figi):
    """
    Получение информации об инструменте по FIGI с кэшированием
    
    Args:
        client: Асинхронный клиент Tinkoff API
        figi: FIGI инструмента
        
    Returns:
        dict: Информация об инструменте
    """
    logger.debug(f"📋 Запрос информации об инструменте FIGI={figi}")
    
    # Проверяем кэш
    if figi in _instrument_cache:
        logger.debug(f"  ✓ Найдено в кэше: {_instrument_cache[figi]['ticker']}")
        return _instrument_cache[figi]
    
    logger.debug(f"  ⏳ Запрос к API Tinkoff...")
    
    try:
        response = await client.instruments.get_instrument_by(
            id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI,
            id=figi
        )
        instrument = response.instrument
        
        info = {
            'name': instrument.name,
            'ticker': instrument.ticker,
            'figi': instrument.figi,
            'currency': instrument.currency,
            'lot': instrument.lot,
            'class_code': instrument.class_code,
            'isin': instrument.isin,
            'type': str(instrument.instrument_type),
        }
        
        # Кэшируем результат
        _instrument_cache[figi] = info
        
        logger.info(f"  ✅ Получена информация: {info['ticker']} ({info['name']})")
        return info
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения информации для FIGI {figi}: {e}")
        
        # Возвращаем fallback данные
        fallback_info = {
            'name': f'Неизвестно ({figi})',
            'ticker': '?',
            'figi': figi,
            'currency': 'RUB',
            'lot': 1,
            'class_code': '?',
            'isin': '?',
            'type': 'unknown'
        }
        
        logger.warning(f"  ⚠️ Используются fallback данные")
        return fallback_info


def clear_instrument_cache():
    """Очистка кэша инструментов"""
    global _instrument_cache
    _instrument_cache.clear()
    logger.info("🗑️ Кэш инструментов очищен")


# ============================================================================
# ПОЛУЧЕНИЕ СЧЕТОВ
# ============================================================================

async def get_accounts(token):
    """
    Получение списка счетов пользователя
    
    Args:
        token: API токен Tinkoff
        
    Returns:
        List[Dict]: Список счетов с полями 'name', 'id', 'type', 'status'
    """
    logger.info("=" * 60)
    logger.info("👤 ПОЛУЧЕНИЕ СПИСКА СЧЕТОВ")
    logger.info("=" * 60)
    
    try:
        async with AsyncClient(token) as client:
            logger.debug("  ⏳ Запрос к API Tinkoff...")
            accounts_response: GetAccountsResponse = await client.users.get_accounts()
            
            accounts = [
                {
                    "name": account.name,
                    "id": account.id,
                    "type": str(account.type),
                    "status": str(account.status)
                }
                for account in accounts_response.accounts
            ]
            
            logger.info(f"✅ Получено счетов: {len(accounts)}")
            for i, acc in enumerate(accounts, 1):
                logger.info(f"  {i}. {acc['name']} (ID: {acc['id'][:8]}...)")
                logger.debug(f"     ├─ Тип: {acc['type']}")
                logger.debug(f"     └─ Статус: {acc['status']}")
            
            logger.info("=" * 60)
            return accounts
            
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ ОШИБКА ПОЛУЧЕНИЯ СПИСКА СЧЕТОВ")
        logger.error("=" * 60)
        logger.error(f"Тип ошибки: {type(e).__name__}")
        logger.error(f"Сообщение: {str(e)}", exc_info=True)
        logger.error("=" * 60)
        raise


# ============================================================================
# ПОЛУЧЕНИЕ ПОРТФЕЛЯ
# ============================================================================

async def get_portfolio(token, account_id):
    """
    Получение информации о портфеле
    
    Args:
        token: API токен Tinkoff
        account_id: ID счета
        
    Returns:
        dict: Информация о портфеле
    """
    logger.info("=" * 60)
    logger.info("💼 ПОЛУЧЕНИЕ ПОРТФЕЛЯ")
    logger.info("=" * 60)
    logger.info(f"  └─ Account ID: {account_id[:8]}...")
    
    try:
        async with AsyncClient(token) as client:
            logger.debug("  ⏳ Запрос к API Tinkoff...")
            portfolio_response: PortfolioResponse = await client.operations.get_portfolio(
                account_id=account_id
            )
            
            portfolio = {
                'account_id': account_id,
                'total_amount': money_value_to_float(portfolio_response.total_amount_portfolio),
                'stocks_amount': money_value_to_float(portfolio_response.total_amount_shares),
                'bonds_amount': money_value_to_float(portfolio_response.total_amount_bonds),
                'etf_amount': money_value_to_float(portfolio_response.total_amount_etf),
                'currencies_amount': money_value_to_float(portfolio_response.total_amount_currencies),
                'futures_amount': money_value_to_float(portfolio_response.total_amount_futures),
                'expected_yield': quotation_to_float(portfolio_response.expected_yield),
                'positions_count': len(portfolio_response.positions)
            }
            
            logger.info("✅ Портфель получен:")
            logger.info(f"  ├─ Общая стоимость: {format_currency(portfolio['total_amount'])}")
            logger.info(f"  ├─ Акции: {format_currency(portfolio['stocks_amount'])}")
            logger.info(f"  ├─ Облигации: {format_currency(portfolio['bonds_amount'])}")
            logger.info(f"  ├─ ETF: {format_currency(portfolio['etf_amount'])}")
            logger.info(f"  ├─ Валюта: {format_currency(portfolio['currencies_amount'])}")
            logger.info(f"  ├─ Ожидаемая доходность: {format_currency(portfolio['expected_yield'])}")
            logger.info(f"  └─ Количество позиций: {portfolio['positions_count']}")
            logger.info("=" * 60)
            
            return portfolio
            
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ ОШИБКА ПОЛУЧЕНИЯ ПОРТФЕЛЯ")
        logger.error("=" * 60)
        logger.error(f"Account ID: {account_id[:8]}...")
        logger.error(f"Тип ошибки: {type(e).__name__}")
        logger.error(f"Сообщение: {str(e)}", exc_info=True)
        logger.error("=" * 60)
        raise


# ============================================================================
# ПОЛУЧЕНИЕ ПОЗИЦИЙ
# ============================================================================

async def get_positions(token, account_id):
    """
    Получение детальной информации о позициях в портфеле
    
    Args:
        token: API токен Tinkoff
        account_id: ID счета
        
    Returns:
        List[dict]: Список позиций с детальной информацией
    """
    logger.info("=" * 60)
    logger.info("📊 ПОЛУЧЕНИЕ ПОЗИЦИЙ ПОРТФЕЛЯ")
    logger.info("=" * 60)
    logger.info(f"  └─ Account ID: {account_id[:8]}...")
    
    try:
        async with AsyncClient(token) as client:
            logger.debug("  ⏳ Запрос портфеля...")
            portfolio_response: PortfolioResponse = await client.operations.get_portfolio(
                account_id=account_id
            )
            
            logger.info(f"  ✓ Найдено позиций: {len(portfolio_response.positions)}")
            
            positions = []
            
            for idx, pos in enumerate(portfolio_response.positions, 1):
                if not pos.figi or pos.quantity is None:
                    logger.debug(f"  ⏭️ Позиция {idx}: пропущена (нет FIGI или quantity)")
                    continue
                
                logger.debug(f"  ⏳ Обработка позиции {idx}/{len(portfolio_response.positions)}...")
                
                # Получаем информацию об инструменте
                instrument_info = await get_instrument_info(client, pos.figi)
                
                current_price = money_value_to_float(pos.current_price)
                avg_price = money_value_to_float(pos.average_position_price)
                quantity = quotation_to_float(pos.quantity)
                expected_yield = money_value_to_float(pos.expected_yield)
                
                total_cost = quantity * current_price
                yield_percent = (expected_yield / (quantity * avg_price) * 100) if avg_price > 0 else 0
                
                position_data = {
                    'name': instrument_info['name'],
                    'ticker': instrument_info['ticker'],
                    'figi': pos.figi,
                    'type': instrument_info['type'],
                    'currency': instrument_info['currency'],
                    'quantity': quantity,
                    'lot': instrument_info['lot'],
                    'average_price': avg_price,
                    'current_price': current_price,
                    'total_cost': total_cost,
                    'expected_yield': expected_yield,
                    'yield_percent': yield_percent,
                    'average_price_formatted': format_currency(avg_price, instrument_info['currency']),
                    'current_price_formatted': format_currency(current_price, instrument_info['currency']),
                    'total_cost_formatted': format_currency(total_cost, instrument_info['currency']),
                    'expected_yield_formatted': format_currency(expected_yield, instrument_info['currency'])
                }
                
                positions.append(position_data)
                
                logger.info(f"  ✓ {len(positions)}. {instrument_info['ticker']}: {quantity} шт. × {format_currency(current_price, instrument_info['currency'])}")
            
            # Сортируем по стоимости позиции (убывание)
            positions = sorted(positions, key=lambda x: x['total_cost'], reverse=True)
            
            logger.info("=" * 60)
            logger.info(f"✅ ПОЗИЦИИ ОБРАБОТАНЫ: {len(positions)}")
            logger.info("=" * 60)
            
            return positions
            
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ ОШИБКА ПОЛУЧЕНИЯ ПОЗИЦИЙ")
        logger.error("=" * 60)
        logger.error(f"Account ID: {account_id[:8]}...")
        logger.error(f"Тип ошибки: {type(e).__name__}")
        logger.error(f"Сообщение: {str(e)}", exc_info=True)
        logger.error("=" * 60)
        raise


# ============================================================================
# ПОЛУЧЕНИЕ ОПЕРАЦИЙ
# ============================================================================

async def get_operations(
    token,
    account_id,
    from_date = None,
    to_date = None,
    figi = None
):
    """
    Получение истории операций
    
    Args:
        token: API токен Tinkoff
        account_id: ID счета
        from_date: Дата начала периода (по умолчанию - месяц назад)
        to_date: Дата окончания периода (по умолчанию - сейчас)
        figi: FIGI инструмента для фильтрации (опционально)
        
    Returns:
        List[dict]: Список операций
    """
    if from_date is None:
        from_date = datetime.now() - timedelta(days=30)
    if to_date is None:
        to_date = datetime.now()
    
    logger.info("=" * 60)
    logger.info("📜 ПОЛУЧЕНИЕ ИСТОРИИ ОПЕРАЦИЙ")
    logger.info("=" * 60)
    logger.info(f"  ├─ Account ID: {account_id[:8]}...")
    logger.info(f"  ├─ Период: {from_date.strftime('%Y-%m-%d')} → {to_date.strftime('%Y-%m-%d')}")
    logger.info(f"  └─ FIGI фильтр: {figi or 'не задан'}")
    
    try:
        async with AsyncClient(token) as client:
            logger.debug("  ⏳ Запрос к API Tinkoff...")
            operations_response: OperationsResponse = await client.operations.get_operations(
                account_id=account_id,
                from_=from_date,
                to=to_date,
                figi=figi
            )
            
            operations = []
            
            for idx, oper in enumerate(operations_response.operations, 1):
                operation_data = {
                    'id': oper.id,
                    'parent_operation_id': oper.parent_operation_id,
                    'currency': oper.currency,
                    'payment': money_value_to_float(oper.payment),
                    'price': money_value_to_float(oper.price),
                    'state': str(oper.state),
                    'quantity': oper.quantity,
                    'quantity_rest': oper.quantity_rest,
                    'figi': oper.figi,
                    'instrument_type': oper.instrument_type,
                    'date': oper.date.isoformat() if oper.date else None,
                    'type': oper.type,
                    'operation_type': str(oper.operation_type),
                }
                
                operations.append(operation_data)
            
            logger.info(f"✅ Получено операций: {len(operations)}")
            logger.info("=" * 60)
            
            return operations
            
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ ОШИБКА ПОЛУЧЕНИЯ ОПЕРАЦИЙ")
        logger.error("=" * 60)
        logger.error(f"Account ID: {account_id[:8]}...")
        logger.error(f"Период: {from_date} → {to_date}")
        logger.error(f"Тип ошибки: {type(e).__name__}")
        logger.error(f"Сообщение: {str(e)}", exc_info=True)
        logger.error("=" * 60)
        raise


# ============================================================================
# ПОИСК ИНСТРУМЕНТОВ
# ============================================================================

async def search_instrument(token, query, instrument_type = None):
    """
    Поиск инструмента по названию или тикеру с фильтрацией по API-доступности 
    и строгим совпадением тикера.
    
    Args:
        token: API токен Tinkoff
        query: Поисковый запрос (тикер или название)
        instrument_type: Тип инструмента для фильтрации (опционально)
        
    Returns:
        List[dict]: Список найденных инструментов
    """
    normalized_query = query.strip().upper()
    logger.info("=" * 60)
    logger.info("🔍 ПОИСК ИНСТРУМЕНТА")
    logger.info("=" * 60)
    logger.info(f"  ├─ Запрос: '{normalized_query}'")
    logger.info(f"  └─ Тип фильтра: {instrument_type or 'все типы'}")
    
    try:
        async with AsyncClient(token) as client:
            logger.debug("  ⏳ Запрос к API Tinkoff...")
            # Используем оригинальный query для поиска, чтобы получить больше результатов
            search_response = await client.instruments.find_instrument(query=query) 
            
            results = []
            
            for idx, instrument in enumerate(search_response.instruments, 1):
                # 1. Фильтрация по типу инструмента (если задано)
                if instrument_type and str(instrument.instrument_type) != instrument_type:
                    logger.debug(f"  ⏭️ {idx}. {instrument.ticker}: пропущен (тип {instrument.instrument_type})")
                    continue
                
                # 2. Строгое совпадение тикера (только если запрос похож на тикер)
                if len(normalized_query) <= 6 and instrument.ticker.upper() != normalized_query:
                    logger.debug(f"  ⏭️ {idx}. {instrument.ticker}: пропущен (тикер не совпадает)")
                    continue
                
                # Получаем полную информацию об инструменте через FIGI
                try:
                    logger.debug(f"  ⏳ Получение полной информации для {instrument.ticker}...")
                    
                    full_instrument_response = await client.instruments.get_instrument_by(
                        id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI,
                        id=instrument.figi
                    )
                    full_instrument = full_instrument_response.instrument
                    
                    # 3. ФИЛЬТРАЦИЯ ПО API ДОСТУПНОСТИ
                    is_api_available = getattr(full_instrument, 'api_trade_available_flag', False)
                    if not is_api_available:
                        logger.info(f"  ❌ {idx}. {full_instrument.ticker} - пропущен (недоступен через API)")
                        continue
                    
                    # --- ИНСТРУМЕНТ ПРОШЕЛ ВСЕ ФИЛЬТРЫ ---
                    
                    result = {
                        'name': full_instrument.name,
                        'ticker': full_instrument.ticker,
                        'figi': full_instrument.figi,
                        'uid': full_instrument.uid,
                        'type': str(full_instrument.instrument_type),
                        'currency': full_instrument.currency,
                        'class_code': full_instrument.class_code,
                        'isin': full_instrument.isin,
                        'lot': full_instrument.lot,
                        'min_price_increment': quotation_to_float(full_instrument.min_price_increment) if hasattr(full_instrument, 'min_price_increment') else None,
                        'trading_status': str(full_instrument.trading_status) if hasattr(full_instrument, 'trading_status') else 'unknown',
                        # Поля доступности
                        'api_trade_available_flag': is_api_available,
                        'buy_available_flag': getattr(full_instrument, 'buy_available_flag', True),
                        'sell_available_flag': getattr(full_instrument, 'sell_available_flag', True),
                        'for_qual_investor_flag': getattr(full_instrument, 'for_qual_investor_flag', False),
                        'for_iis_flag': getattr(full_instrument, 'for_iis_flag', True),
                    }
                    
                    results.append(result)
                    logger.info(f"  ✅ {len(results)}. {full_instrument.ticker} - {full_instrument.name} ({full_instrument.currency})")
                    
                except Exception as e:
                    logger.warning(f"  ⚠️ Не удалось получить полную информацию для {instrument.ticker}: {e}")
                    # В случае ошибки получения полной информации, пропускаем его, так как не можем проверить доступность API
                    continue 
            
            logger.info("=" * 60)
            logger.info(f"✅ НАЙДЕНО ИНСТРУМЕНТОВ: {len(results)}")
            logger.info(f"  └─ Отобраны только те, где Ticker совпадает и API доступен.")
            logger.info("=" * 60)
            
            return results
            
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ ОШИБКА ПОИСКА ИНСТРУМЕНТА")
        logger.error("=" * 60)
        logger.error(f"Запрос: '{query}'")
        logger.error(f"Тип ошибки: {type(e).__name__}")
        logger.error(f"Сообщение: {str(e)}", exc_info=True)
        logger.error("=" * 60)
        raise


# ============================================================================
# ПРОВЕРКА ДОСТУПНОСТИ ИНСТРУМЕНТА ДЛЯ ТОРГОВЛИ
# ============================================================================

async def check_instrument_tradeable(token, instrument):
    """
    Проверка доступности инструмента для торговли
    
    Args:
        token: API токен Tinkoff
        instrument: Словарь с информацией об инструменте
        
    Returns:
        bool: True если инструмент доступен для торговли
    """
    try:
        # Проверяем статус торговли
        trading_status = instrument.get('trading_status', '').lower()
        
        # Список статусов, при которых торговля доступна
        tradeable_statuses = [
            'normal_trading',
            'security_trading_status_normal_trading',
            'trading_status_normal_trading'
        ]
        
        # Проверяем наличие флагов доступности
        api_trade_available = instrument.get('api_trade_available_flag', True)
        buy_available = instrument.get('buy_available_flag', True)
        sell_available = instrument.get('sell_available_flag', True)
        
        is_tradeable = (
            any(status in trading_status for status in tradeable_statuses) and
            api_trade_available and
            (buy_available or sell_available)
        )
        
        logger.debug(f"  🔍 Проверка доступности {instrument.get('ticker')}:")
        logger.debug(f"    ├─ Trading status: {trading_status}")
        logger.debug(f"    ├─ API trade: {api_trade_available}")
        logger.debug(f"    ├─ Buy available: {buy_available}")
        logger.debug(f"    ├─ Sell available: {sell_available}")
        logger.debug(f"    └─ Tradeable: {is_tradeable}")
        
        return is_tradeable
        
    except Exception as e:
        logger.warning(f"  ⚠️ Ошибка проверки доступности: {e}")
        return False


# ============================================================================
# ПОЛУЧЕНИЕ ИНСТРУМЕНТА ПО ТИКЕРУ С ПРОВЕРКОЙ ДОСТУПНОСТИ
# ============================================================================

async def get_instrument_by_ticker(token, ticker):
    """
    Быстрый поиск ДОСТУПНОГО ДЛЯ ТОРГОВЛИ инструмента по тикеру
    
    Args:
        token: API токен Tinkoff
        ticker: Тикер инструмента (например, 'SBER')
        
    Returns:
        dict: Информация об инструменте или None
    """
    logger.info(f"🔍 Поиск доступного для торговли инструмента: {ticker}")
    
    try:
        # Получаем все инструменты по тикеру
        instruments = await search_instrument(token, ticker)
        
        if not instruments:
            logger.warning(f"  ⚠️ Инструмент '{ticker}' не найден")
            return None
        
        logger.info(f"  ✓ Найдено инструментов: {len(instruments)}")
        
        # Ищем точное совпадение по тикеру среди доступных
        tradeable_instruments = []
        
        for instrument in instruments:
            if instrument['ticker'].upper() == ticker.upper():
                logger.debug(f"  🔍 Проверка: {instrument['ticker']} ({instrument.get('name')})")
                
                # Проверяем доступность для торговли
                if await check_instrument_tradeable(token, instrument):
                    tradeable_instruments.append(instrument)
                    logger.info(f"  ✅ Доступен для торговли: {instrument['ticker']}")
                else:
                    logger.warning(f"  ⚠️ НЕ доступен для торговли: {instrument['ticker']}")
        
        # Возвращаем первый доступный инструмент
        if tradeable_instruments:
            selected = tradeable_instruments[0]
            logger.info(f"  ✅ Выбран инструмент: {selected['ticker']} - {selected.get('name')}")
            return selected
        
        # Если точных совпадений нет, возвращаем первый результат
        logger.warning(f"  ⚠️ Точное совпадение не найдено, используется первый результат")
        return instruments[0]
        
    except Exception as e:
        logger.error(f"❌ Ошибка поиска инструмента по тикеру '{ticker}': {e}")
        return None


# ============================================================================
# ПОИСК АКЦИЙ (УПРОЩЁННЫЙ МЕТОД)
# ============================================================================

async def search_shares(token, query):
    """
    Упрощённый поиск только акций
    
    Args:
        token: API токен Tinkoff
        query: Поисковый запрос (тикер или название)
        
    Returns:
        List[dict]: Список найденных акций
    """
    logger.info("=" * 60)
    logger.info("🔍 ПОИСК АКЦИЙ")
    logger.info("=" * 60)
    logger.info(f"  └─ Запрос: '{query}'")
    
    try:
        async with AsyncClient(token) as client:
            logger.debug("  ⏳ Запрос к API Tinkoff (shares)...")
            
            # Используем специальный метод для акций
            shares_response = await client.instruments.shares()
            
            # Фильтруем по запросу
            query_lower = query.lower()
            results = []
            
            for share in shares_response.instruments:
                # Ищем совпадение по тикеру или названию
                if (query_lower in share.ticker.lower() or 
                    query_lower in share.name.lower()):
                    
                    result = {
                        'name': share.name,
                        'ticker': share.ticker,
                        'figi': share.figi,
                        'type': str(share.instrument_type),
                        'currency': share.currency,
                        'class_code': share.class_code,
                        'isin': share.isin,
                        'lot': share.lot,
                        'min_price_increment': quotation_to_float(share.min_price_increment),
                        'trading_status': str(share.trading_status) if hasattr(share, 'trading_status') else 'unknown'
                    }
                    
                    results.append(result)
                    logger.info(f"  ✓ {len(results)}. {share.ticker} - {share.name}")
                    
                    # Ограничиваем количество результатов
                    if len(results) >= 10:
                        break
            
            logger.info("=" * 60)
            logger.info(f"✅ НАЙДЕНО АКЦИЙ: {len(results)}")
            logger.info("=" * 60)
            
            return results
            
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ ОШИБКА ПОИСКА АКЦИЙ")
        logger.error("=" * 60)
        logger.error(f"Запрос: '{query}'")
        logger.error(f"Тип ошибки: {type(e).__name__}")
        logger.error(f"Сообщение: {str(e)}", exc_info=True)
        logger.error("=" * 60)
        raise


# ============================================================================
# РАЗМЕЩЕНИЕ ЗАЯВКИ
# ============================================================================

async def place_order(
    token,
    account_id,
    figi,
    quantity,
    direction, # 'buy' или 'sell'
    order_type, # 'market' или 'limit'
    price = None
):
    """
    Размещение торговой заявки
    
    Args:
        token: API токен Tinkoff
        account_id: ID счета
        figi: FIGI инструмента
        quantity: Количество лотов
        direction: Направление ('buy' или 'sell')
        order_type: Тип заявки ('market' или 'limit')
        price: Цена для лимитной заявки
        
    Returns:
        dict: Информация о размещенной заявке
    """
    logger.info("=" * 80)
    logger.info("💳 РАЗМЕЩЕНИЕ ТОРГОВОЙ ЗАЯВКИ")
    logger.info("=" * 80)
    logger.info(f"  ├─ Account ID: {account_id[:8]}...")
    logger.info(f"  ├─ FIGI: {figi}")
    logger.info(f"  ├─ Направление: {'ПОКУПКА' if direction.lower() == 'buy' else 'ПРОДАЖА'}")
    logger.info(f"  ├─ Количество лотов: {quantity}")
    logger.info(f"  ├─ Тип заявки: {'РЫНОЧНАЯ' if order_type.lower() == 'market' else 'ЛИМИТНАЯ'}")
    logger.info(f"  └─ Цена: {price if price else 'рыночная'}")
    
    try:
        async with AsyncClient(token) as client:
            # Определяем направление
            order_direction = OrderDirection.ORDER_DIRECTION_BUY if direction.lower() == 'buy' else OrderDirection.ORDER_DIRECTION_SELL
            
            # Размещаем заявку
            if price is None:
                logger.debug("  ⏳ Размещение рыночной заявки...")
                
                response = await client.orders.post_order(
                    instrument_id=figi,
                    quantity=quantity,
                    direction=order_direction,
                    account_id=account_id,
                    order_type=OrderType.ORDER_TYPE_MARKET,
                )
            else:
                if price is None:
                    logger.error("❌ Для лимитной заявки необходимо указать цену")
                    raise ValueError("Для лимитной заявки необходимо указать цену")
                
                price_quotation = float_to_quotation(price)
                
                logger.debug(f"  ⏳ Размещение лимитной заявки с ценой {price}...")
                
                response = await client.orders.post_order(
                    instrument_id=figi,
                    quantity=quantity,
                    price=price_quotation,
                    direction=order_direction,
                    account_id=account_id,
                    order_type=OrderType.ORDER_TYPE_LIMIT,
                )
            
            result = {
                'order_id': response.order_id,
                'execution_report_status': str(response.execution_report_status),
                'lots_requested': response.lots_requested,
                'lots_executed': response.lots_executed,
                'initial_order_price': money_value_to_float(response.initial_order_price),
                'executed_order_price': money_value_to_float(response.executed_order_price),
                'total_order_amount': money_value_to_float(response.total_order_amount),
                'initial_commission': money_value_to_float(response.initial_commission),
                'executed_commission': money_value_to_float(response.executed_commission),
                'figi': response.figi,
                'direction': str(response.direction),
                'message': response.message
            }
            
            logger.info("=" * 80)
            logger.info("✅ ЗАЯВКА УСПЕШНО РАЗМЕЩЕНА")
            logger.info("=" * 80)
            logger.info(f"  ├─ Order ID: {result['order_id']}")
            logger.info(f"  ├─ Статус: {result['execution_report_status']}")
            logger.info(f"  ├─ Лотов запрошено: {result['lots_requested']}")
            logger.info(f"  ├─ Лотов исполнено: {result['lots_executed']}")
            logger.info(f"  ├─ Общая сумма: {format_currency(result['total_order_amount'])}")
            logger.info(f"  └─ Сообщение: {result['message']}")
            logger.info("=" * 80)
            
            return result
            
    except ValueError as e:
        logger.error("=" * 80)
        logger.error("❌ ОШИБКА ВАЛИДАЦИИ ПАРАМЕТРОВ ЗАЯВКИ")
        logger.error("=" * 80)
        logger.error(f"Сообщение: {str(e)}")
        logger.error("=" * 80)
        raise
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ ОШИБКА РАЗМЕЩЕНИЯ ЗАЯВКИ")
        logger.error("=" * 80)
        logger.error(f"Тип ошибки: {type(e).__name__}")
        logger.error(f"Сообщение: {str(e)}", exc_info=True)
        logger.error("=" * 80)
        raise


# ============================================================================
# ПОЛУЧЕНИЕ ПОЛНОЙ СВОДКИ ПОРТФЕЛЯ
# ============================================================================

async def get_portfolio_summary(token, account_id):
    """
    Получение полной сводки по портфелю (портфель + позиции)
    
    Args:
        token: API токен Tinkoff
        account_id: ID счета
        
    Returns:
        dict: Полная информация о портфеле
    """
    logger.info("=" * 80)
    logger.info("📊 ПОЛУЧЕНИЕ ПОЛНОЙ СВОДКИ ПОРТФЕЛЯ")
    logger.info("=" * 80)
    logger.info(f"  └─ Account ID: {account_id[:8]}...")
    
    try:
        logger.debug("  ⏳ Запрос информации о портфеле...")
        portfolio_info = await get_portfolio(token, account_id)
        
        logger.debug("  ⏳ Запрос позиций...")
        positions = await get_positions(token, account_id)
        
        summary = {
            'portfolio': portfolio_info,
            'positions': positions,
            'summary': {
                'total_positions': len(positions),
                'total_value': portfolio_info['total_amount'],
                'total_yield': sum(p['expected_yield'] for p in positions),
                'total_yield_percent': (sum(p['expected_yield'] for p in positions) / portfolio_info['total_amount'] * 100) if portfolio_info['total_amount'] > 0 else 0
            }
        }
        
        logger.info("=" * 80)
        logger.info("✅ СВОДКА ПОРТФЕЛЯ ПОЛУЧЕНА")
        logger.info("=" * 80)
        logger.info(f"  ├─ Позиций: {summary['summary']['total_positions']}")
        logger.info(f"  ├─ Общая стоимость: {format_currency(summary['summary']['total_value'])}")
        logger.info(f"  ├─ Общая доходность: {format_currency(summary['summary']['total_yield'])}")
        logger.info(f"  └─ Доходность %: {summary['summary']['total_yield_percent']:.2f}%")
        logger.info("=" * 80)
        
        return summary
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ ОШИБКА ПОЛУЧЕНИЯ СВОДКИ ПОРТФЕЛЯ")
        logger.error("=" * 80)
        logger.error(f"Account ID: {account_id[:8]}...")
        logger.error(f"Тип ошибки: {type(e).__name__}")
        logger.error(f"Сообщение: {str(e)}", exc_info=True)
        logger.error("=" * 80)
        raise

# ============================================================================
# ПОЛУЧЕНИЕ ПОДРОБНОЙ ИНФОРМАЦИИ О ЦЕННОЙ БУМАГЕ
# ============================================================================

async def get_detailed_instrument_info(token, ticker_or_figi, use_figi=False):
    """
    Получение подробной информации об инструменте
    
    Args:
        token: API токен Tinkoff
        ticker_or_figi: Тикер или FIGI инструмента
        use_figi: Если True, ищем по FIGI, иначе по тикеру
        
    Returns:
        dict: Подробная информация об инструменте
    """
    logger.info("=" * 80)
    logger.info("📊 ПОЛУЧЕНИЕ ИНФОРМАЦИИ ОБ ИНСТРУМЕНТЕ")
    logger.info("=" * 80)
    logger.info(f"  ├─ Идентификатор: {ticker_or_figi}")
    logger.info(f"  └─ Поиск по: {'FIGI' if use_figi else 'Тикер'}")
    
    try:
        async with AsyncClient(token) as client:
            # Если ищем по FIGI
            if use_figi:
                figi = ticker_or_figi
            else:
                # Сначала находим инструмент по тикеру
                instrument = await get_instrument_by_ticker(token, ticker_or_figi)
                if not instrument:
                    logger.error(f"  ❌ Инструмент {ticker_or_figi} не найден")
                    return None
                figi = instrument['figi']
            
            logger.info(f"  ⏳ Получение данных по FIGI: {figi}")
            
            # Получаем полную информацию об инструменте
            instrument_response = await client.instruments.get_instrument_by(
                id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI,
                id=figi
            )
            instr = instrument_response.instrument
            
            # Получаем текущую цену (последняя сделка)
            last_prices_response = await client.market_data.get_last_prices(figi=[figi])
            last_price = None
            if last_prices_response.last_prices:
                last_price = quotation_to_float(last_prices_response.last_prices[0].price)
            
            # Получаем стакан заявок (лучшие цены покупки/продажи)
            try:
                orderbook = await client.market_data.get_order_book(
                    figi=figi,
                    depth=10
                )
                best_bid = quotation_to_float(orderbook.bids[0].price) if orderbook.bids else None
                best_ask = quotation_to_float(orderbook.asks[0].price) if orderbook.asks else None
            except Exception as e:
                logger.warning(f"  ⚠️ Не удалось получить стакан: {e}")
                best_bid = None
                best_ask = None
            
            # Получаем свечи (исторические данные за последние 30 дней)
            try:
                from datetime import datetime, timedelta
                now = datetime.utcnow()
                from_date = now - timedelta(days=30)
                
                candles_response = await client.market_data.get_candles(
                    figi=figi,
                    from_=from_date,
                    to=now,
                    interval=CandleInterval.CANDLE_INTERVAL_DAY
                )
                
                candles_data = []
                for candle in candles_response.candles[-30:]:  # Последние 30 дней
                    candles_data.append({
                        'time': candle.time.isoformat(),
                        'open': quotation_to_float(candle.open),
                        'high': quotation_to_float(candle.high),
                        'low': quotation_to_float(candle.low),
                        'close': quotation_to_float(candle.close),
                        'volume': candle.volume,
                    })
                
                # Расчёт изменения за период
                if len(candles_data) >= 2:
                    first_price = candles_data[0]['close']
                    current_price = candles_data[-1]['close']
                    price_change = current_price - first_price
                    price_change_percent = (price_change / first_price) * 100 if first_price else 0
                else:
                    price_change = None
                    price_change_percent = None
                
            except Exception as e:
                logger.warning(f"  ⚠️ Не удалось получить свечи: {e}")
                candles_data = []
                price_change = None
                price_change_percent = None
            
            # Формируем результат
            result = {
                # Основная информация
                'name': instr.name,
                'ticker': instr.ticker,
                'figi': instr.figi,
                'uid': instr.uid,
                'isin': instr.isin,
                'class_code': instr.class_code,
                'instrument_type': str(instr.instrument_type),
                
                # Торговая информация
                'currency': instr.currency,
                'lot': instr.lot,
                'min_price_increment': quotation_to_float(instr.min_price_increment),
                'trading_status': str(instr.trading_status),
                
                # Флаги доступности
                'api_trade_available_flag': getattr(instr, 'api_trade_available_flag', True),
                'buy_available_flag': getattr(instr, 'buy_available_flag', True),
                'sell_available_flag': getattr(instr, 'sell_available_flag', True),
                'for_qual_investor_flag': getattr(instr, 'for_qual_investor_flag', False),
                'for_iis_flag': getattr(instr, 'for_iis_flag', True),
                
                # Рыночные данные
                'last_price': last_price,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': (best_ask - best_bid) if (best_ask and best_bid) else None,
                
                # Исторические данные
                'price_change_30d': price_change,
                'price_change_percent_30d': price_change_percent,
                'candles_30d': candles_data,
                
                # Дополнительная информация
                'sector': getattr(instr, 'sector', None),
                'country_of_risk': getattr(instr, 'country_of_risk', None),
                'country_of_risk_name': getattr(instr, 'country_of_risk_name', None),
                
                # Для акций
                'share_type': str(getattr(instr, 'share_type', None)) if hasattr(instr, 'share_type') else None,
                'ipo_date': getattr(instr, 'ipo_date', None),
                
                # Для облигаций
                'maturity_date': getattr(instr, 'maturity_date', None),
                'coupon_quantity_per_year': getattr(instr, 'coupon_quantity_per_year', None),
                'nominal': money_value_to_float(getattr(instr, 'nominal', None)) if hasattr(instr, 'nominal') else None,
            }
            
            logger.info("=" * 80)
            logger.info("✅ ИНФОРМАЦИЯ ОБ ИНСТРУМЕНТЕ ПОЛУЧЕНА")
            logger.info("=" * 80)
            logger.info(f"  ├─ Название: {result['name']}")
            logger.info(f"  ├─ Тикер: {result['ticker']}")
            logger.info(f"  ├─ Текущая цена: {result['last_price']} {result['currency']}")
            if result['price_change_percent_30d'] is not None:
                logger.info(f"  ├─ Изменение (30д): {result['price_change_percent_30d']:.2f}%")
            logger.info(f"  └─ Статус торговли: {result['trading_status']}")
            logger.info("=" * 80)
            
            return result
            
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ ОШИБКА ПОЛУЧЕНИЯ ИНФОРМАЦИИ ОБ ИНСТРУМЕНТЕ")
        logger.error("=" * 80)
        logger.error(f"Идентификатор: {ticker_or_figi}")
        logger.error(f"Тип ошибки: {type(e).__name__}")
        logger.error(f"Сообщение: {str(e)}", exc_info=True)
        logger.error("=" * 80)
        return None


async def get_multiple_instruments_info(token, tickers):
    """
    Получение информации о нескольких инструментах параллельно
    
    Args:
        token: API токен Tinkoff
        tickers: Список тикеров
        
    Returns:
        dict: Словарь {ticker: info}
    """
    logger.info("=" * 80)
    logger.info("📊 ПОЛУЧЕНИЕ ИНФОРМАЦИИ О НЕСКОЛЬКИХ ИНСТРУМЕНТАХ")
    logger.info("=" * 80)
    logger.info(f"  └─ Тикеры: {', '.join(tickers)}")
    
    tasks = [get_detailed_instrument_info(token, ticker) for ticker in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    instruments_data = {}
    for ticker, result in zip(tickers, results):
        if isinstance(result, Exception):
            logger.error(f"  ❌ {ticker}: {result}")
            instruments_data[ticker] = None
        else:
            instruments_data[ticker] = result
    
    successful = sum(1 for v in instruments_data.values() if v is not None)
    logger.info("=" * 80)
    logger.info(f"✅ ПОЛУЧЕНО ДАННЫХ: {successful}/{len(tickers)}")
    logger.info("=" * 80)
    
    return instruments_data


def format_instrument_info_for_llm(instrument_data):
    """
    Форматирование информации об инструменте для передачи в LLM
    
    Args:
        instrument_data: Данные об инструменте
        
    Returns:
        str: Отформатированная строка
    """
    if not instrument_data:
        return "Информация об инструменте недоступна."
    
    # Статус торговли на русском
    trading_status_map = {
        'SECURITY_TRADING_STATUS_NORMAL_TRADING': '✅ Торгуется',
        'SECURITY_TRADING_STATUS_NOT_AVAILABLE_FOR_TRADING': '❌ Недоступен',
        'SECURITY_TRADING_STATUS_CLOSING_AUCTION': '🔄 Закрывающий аукцион',
        'SECURITY_TRADING_STATUS_OPENING_AUCTION': '🔄 Открывающий аукцион',
        'SECURITY_TRADING_STATUS_BREAK_IN_TRADING': '⏸️ Перерыв в торгах',
    }
    
    trading_status = trading_status_map.get(
        instrument_data.get('trading_status', ''),
        instrument_data.get('trading_status', 'Неизвестно')
    )
    
    formatted = f"""📊 **{instrument_data.get('name')} ({instrument_data.get('ticker')})**

#### Основная информация
- **Тип инструмента:** {instrument_data.get('instrument_type', 'N/A')}
- **FIGI:** `{instrument_data.get('figi')}`
- **ISIN:** `{instrument_data.get('isin')}`
- **Валюта:** {instrument_data.get('currency')}
- **Лот:** {instrument_data.get('lot')} шт.

#### Рыночные данные
"""
    
    if instrument_data.get('last_price'):
        formatted += f"- **Текущая цена:** {instrument_data['last_price']:.2f} {instrument_data.get('currency')}\n"
    
    if instrument_data.get('best_bid') and instrument_data.get('best_ask'):
        formatted += f"- **Лучшая цена покупки (bid):** {instrument_data['best_bid']:.2f}\n"
        formatted += f"- **Лучшая цена продажи (ask):** {instrument_data['best_ask']:.2f}\n"
        formatted += f"- **Спред:** {instrument_data.get('spread', 0):.2f}\n"
    
    if instrument_data.get('price_change_percent_30d') is not None:
        change = instrument_data['price_change_percent_30d']
        emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        formatted += f"- **Изменение за 30 дней:** {emoji} {change:+.2f}%\n"
    
    formatted += f"\n#### Статус торговли\n- {trading_status}\n"
    
    # Доступность для торговли
    if instrument_data.get('api_trade_available_flag'):
        formatted += "- ✅ Доступен для торговли через API\n"
    else:
        formatted += "- ❌ Недоступен для торговли через API\n"
    
    if instrument_data.get('for_qual_investor_flag'):
        formatted += "- ⚠️ Только для квалифицированных инвесторов\n"
    
    # Дополнительная информация для акций
    if instrument_data.get('sector'):
        formatted += f"\n#### Дополнительно\n- **Сектор:** {instrument_data['sector']}\n"
    
    if instrument_data.get('country_of_risk_name'):
        formatted += f"- **Страна риска:** {instrument_data['country_of_risk_name']}\n"
    
    if instrument_data.get('ipo_date'):
        formatted += f"- **Дата IPO:** {instrument_data['ipo_date']}\n"
    
    # Для облигаций
    if instrument_data.get('maturity_date'):
        formatted += f"\n#### Параметры облигации\n"
        formatted += f"- **Дата погашения:** {instrument_data['maturity_date']}\n"
        if instrument_data.get('coupon_quantity_per_year'):
            formatted += f"- **Купонов в год:** {instrument_data['coupon_quantity_per_year']}\n"
        if instrument_data.get('nominal'):
            formatted += f"- **Номинал:** {instrument_data['nominal']}\n"
    
    return formatted


def format_multiple_instruments_for_llm(instruments_data):
    """
    Форматирование информации о нескольких инструментах для LLM
    
    Args:
        instruments_data: Dict {ticker: instrument_data}
        
    Returns:
        str: Отформатированная строка
    """
    if not instruments_data:
        return "Информация об инструментах недоступна."
    
    formatted = "📊 **ИНФОРМАЦИЯ О ЦЕННЫХ БУМАГАХ**\n\n"
    
    for ticker, data in instruments_data.items():
        if data:
            formatted += format_instrument_info_for_llm(data)
            formatted += "\n---\n\n"
        else:
            formatted += f"❌ **{ticker}:** Информация недоступна\n\n"
    
    return formatted